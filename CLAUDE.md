# CLAUDE.md — SPARC Lab Latent-Space Safety Filter

Jakov Marohnic's (Tufts CS undergrad) research project in Prof. Ryan Cosner's SPARC Lab, Fall 2026.
Proposal: "Jakov's Fall26 Project Proposal" (in the claude.ai project "Sparc Lab World Model").

## Goal in one paragraph

Reproduce Andrea Bajcsy's latent-space safety filters, then add probabilistic safety guarantees.
An encoder maps images to a latent `z`; a latent world model predicts `z'` from history + action;
a smooth margin `l(z)` (> 0 safe, < 0 failed) is learned from labeled data; an actor-critic learns a
discounted HJ safety value `V(z)` / `Q(z,a)` that acts as a **LatentCBF**; at run time a sampling-based
filter picks the action closest to the nominal policy that satisfies a discrete-time CBF condition.
The novel contribution is to wrap this in **discrete-time stochastic CBF theory** (Cosner, Mestres et al.)
using particle/shooting estimates of safety probability, so the filter comes with a finite-horizon
P(safe) bound instead of no guarantee.

Roadmap: (1) simulated inverted pendulum ← **current**, (2) Dubins car (matches the papers' benchmark
so results are directly comparable), (3) hardware: Unitree D1 arm holding an hourglass with the
constraint "never let the sand run out" (must flip it every Δt). Possible venue: ICRA (Sept deadline).
Expectations are explicitly low-pressure (coursework first; learning the research process = success).

## Key papers (what to take from each)

**[1] Nakamura, Peters, Bajcsy — "Generalizing Safety Beyond Collision-Avoidance via Latent-Space
Reachability Analysis" (arXiv 2502.00935).** Original *Latent Safety Filters* (LatentSafe).
- World model: DreamerV3 RSSM (sim, 128×128 RGB + heading; manipulation adds wrist cam + proprio),
  DINO-WM with frozen DINOv2 + ViT predictor on hardware (H=3 history frames).
- Failure margin = classifier with hinge loss at ±δ, wrapped in tanh. Hardware labels: a human marks
  the first failing frame of each trajectory (~2.5 h for 1,300 trajectories).
- Discounted latent HJ backup: `V(z) = (1-γ) l(z) + γ min{ l(z), max_a E[V(z')] }`, γ = 0.9999.
  Solved with RL in imagination (DDQN for Dubins, DDPG for manipulation); imagined rollouts start from
  encoded dataset observations and are kept short (25 steps) to limit model drift.
- Filter: least-restrictive switching — use π_task if `V(ẑ_{t+1}) > ε` (ε = 0.3–0.4) else π_safe.
- Dubins results: F1 0.984 vs 0.987 for privileged-state RL. Limitations: no formal guarantees, only
  guards against what the world model can predict, OOD dynamics (M&Ms bag) break it.

**[2] Nakamura et al. — "How to Train Your Latent Control Barrier Function: Smooth Safety Filtering
Under Hard-to-Model Constraints" (arXiv 2511.18606).** *LatentCBF* — the method we reimplement.
- Theorem: `L_V ≤ L_l · max{1, (1-γ)/(1-γ L_f)}` → a smooth value needs a smooth margin.
- Margin loss (WGAN-GP style), with `ẑ = η z⁻ + (1-η) z⁺`, η ~ U(0,1):
  `L = λ_w (E_fail[l] - E_safe[l]) + λ_gp E_ẑ[(‖∇l(ẑ)‖ - β)²] + λ_sign · hinge_{δ=0}`
  Dubins: λ_w = 0.1, λ_sign = 1, λ_gp = 10, β = 0.1, MLP [512,512] SiLU. (NoGP baseline: hinge δ = 0.75.)
  The δ=0 sign term pins the zero level set; WGAN alone only fixes l up to a constant.
- Critic target: `y = (1-γ) l + γ min{ l, Q(z', a') }`, l = tanh(ℓ(z)), γ = 0.995; actor loss −Q(z, π(z)).
  Deterministic actor, single critic, [512]×3 LayerNorm/ReLU, lr actor 1e-4 / critic 3e-4, batch 512,
  buffer 100k, 120k iters, imagination horizon T = 8.
- **Data mixing**: replay is 50% safety-policy rollouts, 50% nominal-policy rollouts; each tuple
  (z, a, l, z', a') takes a and a' from the same policy. Without it Q is wrong exactly on the
  nominal actions the filter queries (NoMix → erratic actions/spills).
- Filter: `a* = argmin ‖a − π_nom‖ s.t. Q(z,a) − ε ≥ α (Q(z, π_safe(z)) − ε)`, α ∈ [0,1).
  Solved by sampling: candidates from a mix of π_nom and π_safe (Dubins: 25 evenly spaced actions +
  both policies), keep those satisfying the constraint, return the one closest to π_nom, fall back to
  π_safe if none. Model-free (uses Q directly), ~10 ms for 7,600 samples on hardware.
- Results: GP margin max-gradient 1.2 → 0.17; 45% smaller overrides than least-restrictive at equal
  safety (Dubins); hardware success 38% → 80% with no failures. **No pendulum experiment** in either
  Bajcsy paper — the pendulum is our own warm-up.

**[3] Cosner et al. — "Robust Safety under Stochastic Uncertainty with Discrete-Time Control Barrier
Functions" (RSS 2023, arXiv 2302.07469).**
- `x_{k+1} = F(x_k,u_k) + d_k`, d_k i.i.d., possibly unbounded. DTCBF in expectation:
  `E[h(x_{k+1}) | x_k] ≥ α h(x_k)`.
- Ville's inequality on a supermartingale built from h (requires h ≤ M) bounds the K-step exit
  probability; bound is finite-horizon only and conservative vs Monte Carlo.
- Jensen gap: certainty-equivalent constraints `h(F + E[d]) ≥ α h(x)` are not enough for concave h;
  correct with `c_J = (λ_max/2) tr(cov d)`. Has an inverted pendulum example (|θ| ≤ π/6).

**[4] Cosner, Culbertson, Ames — "Bounding Stochastic Safety: Leveraging Freedman's Inequality with
DTCBFs" (arXiv 2403.05745).** Freedman-based bound `P_u ≤ H(λ/δ, σ√K/δ)` with λ = α^K h(x₀);
needs only bounded one-step error δ and conditional variance σ², not an upper bound on h.
For L_h-Lipschitz h and ‖d‖ ≤ d_max: δ = 2 L_h d_max, σ² = L_h² d_max².

**[5] Mestres, Werner, Cosner, Ames — "Probabilistic Control Barrier Functions: Safety in Probability
for Discrete-Time Stochastic Systems" (arXiv 2510.01501).** The "Cosner & Mestres" PCBF work.
- δ-probabilistic CBF: ∀x ∈ C ∃u with `P(h(x') ≥ α h(x)) ≥ 1 − δ`.
- Main result: if enforced each step, `P(safe for H steps) ≥ (1 − δ)^H`; choose
  δ ≤ 1 − (1 − ε)^{1/H} for ε-safety.
- Sample-based verification of the chance constraint: Markov, Cantelli (mean/variance), Hoeffding,
  and scenario (`N ≥ (2/δ)(ln(1/β) + d)`, distribution-free, needs convexity in u). With per-step
  confidence 1−β, overall confidence ≥ 1 − Hβ. This is the bridge to our particle-based filter:
  sample latent next-states / particles, check the CBF inequality empirically, certify via these bounds.

**Open research question:** the guarantees in [3–5] assume a known disturbance model and a true h.
In our setting h ≈ learned V on learned latents, and "disturbance" ≈ world-model error. Figuring out
how to characterise that error (e.g. ensembles / stochastic latent dynamics / residuals on held-out
data) so the PCBF bound is meaningful is the core novelty.

## Repo layout (flat, run everything from repo root)

| File | Role |
|---|---|
| `enviornment.py` | Gymnasium pendulum env, 84×84 RGB obs. **Filename is misspelled on purpose — imports depend on it; don't rename casually.** State (θ, ω), dt 0.02, m 5, l 1, damping 0.1, \|u\| ≤ 10, failure \|θ\| > `FAIL_ANGLE` = 0.4. With this torque limit the pole can't be held from rest past ~0.2 rad, so the viable set is strictly smaller than the constraint set. |
| `data_collectionINVP.py` | Collects 100 episodes × 200 steps for each policy in `POLICY_TYPES` = pd, noisy_pd, random, pd_then_random. Rolls past failure until \|θ\| ≥ 0.6. Saves `data/trajectories/<policy>_policy.npz` (obs, actions, theta, omega, safe, episode_starts, action_starts; T+1 frames and T actions per episode). |
| `data_utils.py` | `load_split`: train/val split **by whole episode** (frame-level splits leak). |
| `latent_encoder.py` | `Encoder` (CNN → 16-d latent), `Decoder`, `Dynamics` (MLP on a `WINDOW`=3 latent history + action → next latent), `Margin` (MLP, output `bound·tanh(x/bound)`, bound 2), `WorldModel`. |
| `train_WM.py` | Stage 1a: autoencoder, brightness-weighted MSE, keeps best-val checkpoint → `encoder.pt`, `decoder.pt`. |
| `train_dynamics.py` | Stage 1b: frozen encoder, trains `Dynamics` with MSE in latent space → `dynamics.pt`. (Uses all data, no val split.) |
| `train_margin.py` | Stage 2: margin l(z). Flags for hinge ±1, WGAN rank term, GP (two- or one-sided, target β), `--run-name` writes `data/runs/<name>.json`. Prints nearest-safe distances (hinge ±1 with slope β needs gap ≥ 2/β). |
| `train_value.py` | Stage 3: TD3-style (twin critics, target smoothing, delayed actor) with the LatentCBF target, 50/50 mix of imagined safety rollouts and real nominal transitions, γ annealed 0.9 → 0.99. Saves `value_actor.pt` / `value_critic.pt`. |
| `value_function.py` | `Actor`, twin `Critic`, `safety_value(actor, critic, w) = Q_min(w, π(w))`. RL state = window of 3 latents (one frame shows θ but not ω). |
| `ground_truth.py` | Value iteration on the true (θ, ω) grid → viability kernel. The answer key for the learned V. |
| `visualize*.py` | Reconstructions, dynamics rollouts, margin, and learned V vs ground truth (sign agreement, unsafe-called-safe rate). |
| `data/models/` | Checkpoints: encoder, decoder, dynamics, margin, margin_paper, margin_smoke (`.pt`). |
| `data/runs/` | JSON configs + metrics of runs (e.g. `paper.json` = margin with paper hyperparameters). |

Pipeline: `make collect` → `make train` → `make train_dynamics` → `make train_margin` →
`make train_value` → `make visualize_value` (pass extra flags by calling the scripts directly).
`make collect_demo` writes a small held-out noisy-PD set to `data/demo/` (seed 1).

Stack: Python 3.10, PyTorch (CUDA if available, else CPU), Gymnasium, NumPy, PIL, matplotlib.

## Conventions

- Sign convention everywhere: **> 0 safe, < 0 unsafe** (margin, Q, V). Positive torque = positive θ̈.
- Actions are normalised by `A_MAX` before entering the critic; the actor outputs `A_MAX·tanh`.
- `actions[action_starts[i] + t]` moves frame `episode_starts[i] + t` to frame `+ t + 1`. Keep this
  indexing; the windowing code in `train_dynamics.py` / `train_value.py` depends on it.
- Safety labels come from the true state (|θ| ≤ 0.4), not from episode termination.
- Every evaluation of a learned safety object should be compared against `ground_truth.py`; report the
  dangerous error (unsafe-called-safe) separately from the conservative one.
- Prefer adding CLI flags + `--run-name` JSON logs over editing constants, so runs are reproducible.

## Where the code differs from the papers (know before "fixing")

- Margin: the proposal writes hinge ±1 + GP with target 1; the paper uses δ=0 sign + WGAN + GP with β=0.1.
  `train_margin.py` supports both; `data/runs/paper.json` uses the paper's λ's (λ_rank 0.1, λ_gp 10, β 0.1)
  and gets 98% val acc but only 67% accuracy in the |θ| ∈ [0.35, 0.45] band — single-frame latents
  near the boundary overlap (min failed→safe latent distance is 0).
- Value: the proposal's eq. (4) omits the `(1-γ) l` term; the code follows the paper's target.
  Code uses TD3 twin critics (paper: single critic), γ 0.99 (paper 0.995), horizon 50 (paper 8),
  and l bounded by 2·tanh(·/2) rather than tanh.
- World model is a small deterministic CNN autoencoder + MLP dynamics, not an RSSM / DINO-WM. A
  stochastic or ensembled model will probably be needed for the probabilistic guarantees.

## Not built yet (next steps)

1. The LatentCBF safety filter itself (sampling-based: candidates from π_nom/π_safe/grid, constraint
   `Q(z,a) − ε ≥ α (Q(z, π_safe(z)) − ε)`, closest to π_nom, fallback π_safe) + closed-loop eval in
   the real env vs no filter and least-restrictive switching (metrics: safety rate, mean |Δa|, task success).
2. Probabilistic layer: particle/shooting estimate of `P(V(z') ≥ α V(z))`, certified with
   Hoeffding/scenario bounds from [5], and a finite-horizon `(1−δ)^H` claim checked by Monte Carlo
   in the true pendulum (with injected noise).
3. Dubins car env + reproduction of [1]/[2] numbers.
4. Hardware: Unitree D1 + hourglass.

## Lab logistics

Weekly Monday meetings (alternating 30-min 1:1 and 1-h lab paper discussion). Prof. Cosner's research
office hours Fri 3–5 pm, 574 Boston Ave, Room 313.
