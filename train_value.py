"""Stage 3: learn the safety value function and safety policy with TD3-style actor-critic.

Critic target (Smooth LatentCBF, eq. 7):
    y = (1 - gamma) * l(z) + gamma * min{ l(z), Q_target(z', a') }
where l is the margin on the newest frame of the window.

Replay data is mixed half and half (the paper's fix for distribution mismatch):
  * safety half:  rollouts imagined in the world model, actions from pi_safe + exploration noise,
                  a' = pi_safe_target(z') at update time
  * nominal half: real transitions from the collected datasets, a and a' both from the policy that
                  collected them (or a' = pi_safe_target(z') with --nominal-next-action actor)
"""
import argparse
import copy
import json
import os
import numpy as np
import torch
import torch.nn.functional as F
from latent_encoder import Encoder, Dynamics, Margin, WINDOW
from value_function import Actor, Critic, A_MAX, safety_value
from data_collectionINVP import POLICY_TYPES

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LATENT_SIZE = 16
ENCODE_BATCH = 256


def parse_args():
    p = argparse.ArgumentParser(description="Train the latent safety value function (Stage 3).")
    p.add_argument("--margin", default="data/models/margin.pt", help="margin weights to use as l(z)")
    p.add_argument("--margin-bound", type=float, default=2.0, help="must match the margin's training")
    p.add_argument("--steps", type=int, default=100_000, help="gradient steps")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--nominal-fraction", type=float, default=0.5, help="share of each batch from real data")
    p.add_argument("--nominal-next-action", choices=["stored", "actor"], default="stored",
                   help="a' for nominal transitions: the policy's own next action (paper) or pi_safe")
    p.add_argument("--n-envs", type=int, default=256, help="parallel imagined rollouts")
    p.add_argument("--horizon", type=int, default=50, help="imagined steps before a rollout restarts")
    p.add_argument("--warmup-steps", type=int, default=20, help="env steps with uniform random actions")
    p.add_argument("--buffer-size", type=int, default=200_000)
    p.add_argument("--gamma", type=float, default=0.99, help="final discount")
    p.add_argument("--gamma-start", type=float, default=0.9, help="annealed to --gamma over the first half")
    p.add_argument("--lr-actor", type=float, default=1e-4)
    p.add_argument("--lr-critic", type=float, default=3e-4)
    p.add_argument("--tau", type=float, default=0.005, help="target network soft-update rate")
    p.add_argument("--explore-std", type=float, default=0.2, help="exploration noise, fraction of A_MAX")
    p.add_argument("--target-noise", type=float, default=0.1, help="TD3 target smoothing, fraction of A_MAX")
    p.add_argument("--target-noise-clip", type=float, default=0.25, help="fraction of A_MAX")
    p.add_argument("--policy-delay", type=int, default=2)
    p.add_argument("--data-dir", default="data/trajectories")
    p.add_argument("--out-prefix", default="data/models/value", help="saves <prefix>_actor.pt / _critic.pt")
    p.add_argument("--run-name", default=None, help="if set, write config + metrics to data/runs/value_<name>.json")
    p.add_argument("--log-every", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def load_frozen(module, path):
    module.load_state_dict(torch.load(path, weights_only=True, map_location=DEVICE))
    module.to(DEVICE).eval()
    for param in module.parameters():
        param.requires_grad_(False)
    return module


@torch.no_grad()
def build_nominal_transitions(encoder, margin, data_dir, window):
    """Real transitions from the collected datasets.

    Window w_t holds the latents of frames [t - window + 1, ..., t]. actions[t] moves frame t to t + 1.
    Each transition is (w_t, a_t, l(z_t), w_{t+1}, a_{t+1}), valid for t in [window - 1, T - 2].
    Also returns every real window, used as start states for imagined rollouts.
    """
    W, A, W2, A2, starts = [], [], [], [], []
    for name in POLICY_TYPES:
        data = np.load(f"{data_dir}/{name}_policy.npz")
        fs, as_ = data["episode_starts"], data["action_starts"]
        obs, acts = data["obs"], data["actions"]
        for i in range(len(fs) - 1):
            ep_obs = obs[fs[i]:fs[i + 1]]                                              # (T+1, 84, 84, 3)
            ep_act = torch.from_numpy(acts[as_[i]:as_[i + 1]]).float().to(DEVICE)       # (T, 1)
            T = len(ep_act)
            if T + 1 < window:
                continue
            z = torch.cat([encoder(torch.from_numpy(ep_obs[b:b + ENCODE_BATCH]).to(DEVICE))
                           for b in range(0, len(ep_obs), ENCODE_BATCH)])               # (T+1, L)
            win = z.unfold(0, window, 1).permute(0, 2, 1)   # (T+2-window, window, L); win[k] ends at frame k+window-1
            starts.append(win)

            n = T - window   # k = t - window + 1 runs over [0, T - window - 1]
            if n >= 1:
                k = torch.arange(n, device=DEVICE)
                W.append(win[k])
                W2.append(win[k + 1])
                A.append(ep_act[k + window - 1])
                A2.append(ep_act[k + window])

    w = torch.cat(W)
    nominal = {"w": w, "a": torch.cat(A), "l": margin(w[:, -1]), "w2": torch.cat(W2), "a2": torch.cat(A2)}
    return nominal, torch.cat(starts)


class ReplayBuffer:
    """Ring buffer on DEVICE for imagined (safety-policy) transitions."""
    def __init__(self, capacity, window, latent_size):
        self.capacity, self.ptr, self.size = capacity, 0, 0
        self.w = torch.zeros(capacity, window, latent_size, device=DEVICE)
        self.a = torch.zeros(capacity, 1, device=DEVICE)
        self.l = torch.zeros(capacity, device=DEVICE)
        self.w2 = torch.zeros(capacity, window, latent_size, device=DEVICE)

    def add(self, w, a, l, w2):
        idx = (self.ptr + torch.arange(len(w), device=DEVICE)) % self.capacity
        self.w[idx], self.a[idx], self.l[idx], self.w2[idx] = w, a, l, w2
        self.ptr = (self.ptr + len(w)) % self.capacity
        self.size = min(self.size + len(w), self.capacity)

    def sample(self, n):
        idx = torch.randint(self.size, (n,), device=DEVICE)
        return self.w[idx], self.a[idx], self.l[idx], self.w2[idx]


class ImaginedRollouts:
    """n_envs parallel rollouts inside the world model, (re)started from real windows.

    A rollout restarts after `horizon` steps, or one step after its newest frame is classified as
    failed (so the transition out of a failed state is stored once, then the rollout resets)."""
    def __init__(self, dynamics, margin, start_windows, n_envs, horizon):
        self.dynamics, self.margin, self.starts, self.horizon = dynamics, margin, start_windows, horizon
        self.w = self._sample_starts(n_envs)
        self.t = torch.zeros(n_envs, dtype=torch.long, device=DEVICE)

    def _sample_starts(self, n):
        return self.starts[torch.randint(len(self.starts), (n,), device=DEVICE)].clone()

    @torch.no_grad()
    def step(self, actions):
        l = self.margin(self.w[:, -1])
        z_next = self.dynamics(self.w, actions)
        w2 = torch.cat([self.w[:, 1:], z_next.unsqueeze(1)], dim=1)
        transition = (self.w.clone(), actions, l, w2)

        self.t += 1
        reset = (l < 0) | (self.t >= self.horizon)
        self.w = w2.clone()
        n_reset = int(reset.sum())
        if n_reset:
            self.w[reset] = self._sample_starts(n_reset)
            self.t[reset] = 0
        return transition, float((l < 0).float().mean())


def soft_update(target, source, tau):
    with torch.no_grad():
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.mul_(1 - tau).add_(tau * sp)


def gamma_at(step, args):
    frac = min(1.0, step / max(1, args.steps // 2))
    return args.gamma_start + frac * (args.gamma - args.gamma_start)


def smoothed_target_action(actor_target, w2, args):
    a = actor_target(w2)
    noise = (torch.randn_like(a) * args.target_noise * A_MAX).clamp(
        -args.target_noise_clip * A_MAX, args.target_noise_clip * A_MAX)
    return (a + noise).clamp(-A_MAX, A_MAX)


@torch.no_grad()
def diagnostics(actor, critic, nominal):
    """Sanity checks on real data. The target satisfies y <= l(z), so a correct critic has V <= l."""
    n = min(20_000, len(nominal["w"]))
    w, l = nominal["w"][:n], nominal["l"][:n]
    V = safety_value(actor, critic, w)
    return {
        "V_mean": float(V.mean()),
        "V_above_l": float(F.relu(V - l).mean()),        # should approach 0
        "frac_V_neg": float((V < 0).float().mean()),
        "frac_l_neg": float((l < 0).float().mean()),     # V < 0 should be at least this common
    }


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    print(f"Device: {DEVICE}")

    encoder = load_frozen(Encoder(latent_size=LATENT_SIZE), "data/models/encoder.pt")
    dynamics = load_frozen(Dynamics(latent_size=LATENT_SIZE), "data/models/dynamics.pt")
    margin = load_frozen(Margin(latent_size=LATENT_SIZE, bound=args.margin_bound), args.margin)

    print("Encoding datasets...")
    nominal, start_windows = build_nominal_transitions(encoder, margin, args.data_dir, WINDOW)
    n_nominal = len(nominal["w"])
    print(f"{n_nominal} nominal transitions, {len(start_windows)} start windows, "
          f"{(nominal['l'] < 0).float().mean():.1%} of nominal states have l < 0")

    actor, critic = Actor(LATENT_SIZE).to(DEVICE), Critic(LATENT_SIZE).to(DEVICE)
    actor_target, critic_target = copy.deepcopy(actor), copy.deepcopy(critic)
    for net in (actor_target, critic_target):
        for param in net.parameters():
            param.requires_grad_(False)
    opt_actor = torch.optim.Adam(actor.parameters(), lr=args.lr_actor)
    opt_critic = torch.optim.Adam(critic.parameters(), lr=args.lr_critic)

    buffer = ReplayBuffer(args.buffer_size, WINDOW, LATENT_SIZE)
    envs = ImaginedRollouts(dynamics, margin, start_windows, args.n_envs, args.horizon)

    for _ in range(args.warmup_steps):
        a = (torch.rand(args.n_envs, 1, device=DEVICE) * 2 - 1) * A_MAX
        transition, _ = envs.step(a)
        buffer.add(*transition)

    n_nom = int(round(args.batch_size * args.nominal_fraction))
    n_safe = args.batch_size - n_nom
    log = {"critic_loss": 0.0, "actor_loss": 0.0, "q": 0.0, "fail_rate": 0.0, "abs_a": 0.0}
    actor_updates = 0

    for step in range(1, args.steps + 1):
        # 1) one imagined env step with the current safety policy + exploration noise
        with torch.no_grad():
            a_env = actor(envs.w)
            a_env = (a_env + torch.randn_like(a_env) * args.explore_std * A_MAX).clamp(-A_MAX, A_MAX)
        transition, fail_rate = envs.step(a_env)
        buffer.add(*transition)

        # 2) critic update on a mixed batch
        w_s, a_s, l_s, w2_s = buffer.sample(n_safe)
        idx = torch.randint(n_nominal, (n_nom,), device=DEVICE)
        w_n, a_n, l_n, w2_n, a2_n = (nominal[k][idx] for k in ("w", "a", "l", "w2", "a2"))

        gamma = gamma_at(step, args)
        with torch.no_grad():
            a2_s = smoothed_target_action(actor_target, w2_s, args)
            if args.nominal_next_action == "actor":
                a2_n = smoothed_target_action(actor_target, w2_n, args)
            w, a, l = torch.cat([w_s, w_n]), torch.cat([a_s, a_n]), torch.cat([l_s, l_n])
            w2, a2 = torch.cat([w2_s, w2_n]), torch.cat([a2_s, a2_n])
            q_next = critic_target.q_min(w2, a2)
            y = (1 - gamma) * l + gamma * torch.minimum(l, q_next)

        q1, q2 = critic(w, a)
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        opt_critic.zero_grad()
        critic_loss.backward()
        opt_critic.step()

        # 3) delayed actor update: pick actions the critic thinks are safest
        if step % args.policy_delay == 0:
            actor_loss = -critic(w, actor(w))[0].mean()
            opt_actor.zero_grad()
            actor_loss.backward()
            opt_actor.step()
            soft_update(actor_target, actor, args.tau)
            soft_update(critic_target, critic, args.tau)
            log["actor_loss"] += actor_loss.item()
            actor_updates += 1

        log["critic_loss"] += critic_loss.item()
        log["q"] += q1.mean().item()
        log["fail_rate"] += fail_rate
        log["abs_a"] += a_env.abs().mean().item()   # how hard the current safety policy pushes

        if step % args.log_every == 0:
            d = diagnostics(actor, critic, nominal)
            print(f"step {step:>7}  gamma {gamma:.4f}  critic {log['critic_loss'] / args.log_every:.4f}  "
                  f"actor {log['actor_loss'] / max(1, actor_updates):.4f}  Q {log['q'] / args.log_every:+.3f}  "
                  f"imagined fail {log['fail_rate'] / args.log_every:.3f}  |a| {log['abs_a'] / args.log_every:.2f}  "
                  f"| real: V {d['V_mean']:+.3f}  V>l {d['V_above_l']:.4f}  "
                  f"V<0 {d['frac_V_neg']:.3f} (l<0 {d['frac_l_neg']:.3f})")
            log = {k: 0.0 for k in log}
            actor_updates = 0

    os.makedirs(os.path.dirname(args.out_prefix) or ".", exist_ok=True)
    torch.save(actor.state_dict(), f"{args.out_prefix}_actor.pt")
    torch.save(critic.state_dict(), f"{args.out_prefix}_critic.pt")
    print(f"Saved {args.out_prefix}_actor.pt and {args.out_prefix}_critic.pt")

    if args.run_name:
        os.makedirs("data/runs", exist_ok=True)
        path = f"data/runs/value_{args.run_name}.json"
        with open(path, "w") as f:
            json.dump({"config": vars(args), "final": diagnostics(actor, critic, nominal)}, f, indent=2)
        print(f"Metrics written to {path}")


if __name__ == "__main__":
    main()
