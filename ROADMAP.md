# ROADMAP: DINO world model → physical pendulum → Unitree D1

How we go from the current CNN pendulum pipeline to a replication of LatentCBF's hardware setup
(frozen DINO encoder + DINO-WM dynamics), first in simulation, then on a physical pendulum, then on
the D1 arm with the hourglass task. Companion to `CLAUDE.md` (paper notes + repo guide).

Written 2026-11-23. Update the **Status** lines as phases finish.

---

## Guiding rules

1. **Change one thing at a time, measured against ground truth while we still have it.** The sim
   pendulum (`ground_truth.py`) is the only place we can measure exactly how wrong V is. Every
   architecture change gets validated there before it touches hardware.
2. **Hyperparameters live in seconds, not steps.** The paper's settings (imagination horizon 8, γ 0.995) are
   tuned for 10–15 Hz. Convert using our control period `dt`:
   - imagination horizon `T = round(T_sec / dt)` with `T_sec ≈ 0.5–1 s` (pendulum at dt 0.02 → 25–50 steps)
   - discount `γ = 1 − dt / τ` with lookahead `τ ≈ 2–4 s` (dt 0.02, τ 2 s → γ 0.99)
   (Learned the hard way: horizon 8 at dt 0.02 → sign agreement 86.6% → 67.9%, run `value_h8`.)
3. **The learned filter is never the only safety layer on hardware.** Torque/velocity/joint limits,
   an e-stop, and a watchdog run below it, independent of any network.
4. **Keep the CNN pipeline working** as the baseline; DINO code goes in new files, selected by a flag.

---

## Target architecture (what the paper ran on hardware)

| Piece | LatentCBF hardware | Ours |
|---|---|---|
| Encoder | Frozen **DINOv3 ViT-S+/16** (`facebook/dinov3-vits16plus-pretrain-lvd1689m`, 29M params, 384-d, patch 16). 256×256 image → 16×16 = 256 patch tokens × 384 | Same model. DINOv3 weights need the Meta license accepted on Hugging Face; supported in `transformers` ≥ the DINOv3 release (Aug 2025). |
| Latent | 2 camera views + 10-d action embedding + 8-d proprio, concatenated per patch → [256, 786] | Pendulum: 1 view (+ optional proprio, see Phase 3). D1: scene + wrist cam + joint state. |
| Dynamics | DINO-WM: ViT predictor (6 layers, 16 heads, MLP 2048), history H = 3, teacher forcing, loss ‖E(o_{t+1}) − p(z_{t−H:t}, a_{t−H:t})‖, AdamW lr 5e-5, 100k iters | Same design, scaled down first. Reference code: DINO-WM repo `gaoyuezhou/dino_wm` (MIT). Our `WINDOW = 3` already matches H = 3. |
| Margin | Average-pool patch tokens → 2-layer MLP, LayerNorm, ReLU. λ_w 0.2, λ_sign 100, λ_gp 10, β 0.02 | Same; start from these, retune β on sim ground truth. |
| Actor / critic | [512]×3, LayerNorm, ReLU, lr 1e-4 / 3e-4, batch 512, buffer 100k, 120k iters, 50/50 nominal mix, action chunks of 16 | Pool tokens first (paper doesn't say how; pooling is our assumption). Keep TD3 twin critics unless they cause problems. |
| Filter | 7,600 sampled actions scored with Q (no world-model rollouts), ~10 ms; α 0.85–0.925 | Same Q-only sampling; pendulum needs far fewer samples (1-D action). |

LatentCBF's own code was listed as "coming soon" on the project page (cmu-intentlab.github.io/latent_cbf)
when we checked. Check again before writing Phase 1B from scratch.

---

## Phase 0 — Finish the CNN baseline (prerequisite)

Status: **in progress** (value run `omega3` with OMEGA0_RANGE = 3 pending)

- [ ] Value function with acceptable ground-truth agreement (target: sign agreement ≥ 90%, unsafe-called-safe ≈ 0).
- [ ] `safety_filter.py`: LatentCBF filter + least-restrictive baseline + closed-loop eval in the sim env
      (safety rate, mean |Δa|, task success). This filter code is reused unchanged in every later phase.
- [ ] Save `visualize_value.py`'s summary numbers into the run JSON, so runs can be compared without screenshots.

Exit gate: filter beats least-restrictive on |Δa| at equal safety in sim. This is the result to reproduce on hardware.

---

## Phase 1 — Swap the CNN for DINO, in simulation

Goal: same pendulum, same ground truth, DINO features instead of the CNN. Answers "do frozen
pretrained features keep the safety-relevant information?" before any hardware is involved.

### 1A. Pooled DINO vector (small change)
- New `dino_encoder.py`: wraps DINOv3 ViT-S+/16 (frozen, eval, fp16 on GPU). Upsample 84×84 renders
  to 256×256 (or re-render at 256 — preferred; tiny objects lose detail when upsampled).
  Output: mean-pooled patch tokens (384-d) or CLS token, as a flag. Try both.
- New `cache_features.py`: encode every dataset frame once and save features to disk. The encoder is frozen, so
  nothing downstream should re-run DINO during training.
- Make `latent_size` a parameter everywhere instead of the hard-coded 16 (`train_dynamics.py`,
  `train_margin.py`, `train_value.py`, `visualize_value.py`, `value_function.py`), and add a
  `--encoder {cnn,dino_pooled,dino_tokens}` flag.
- `train_WM.py` / decoder are **not needed** for DINO (no reconstruction).
- Retrain dynamics → margin → value on the cached features, compare to the CNN baseline with `visualize_value.py`.

### 1B. Patch tokens + DINO-WM predictor (the paper's actual setup)
- Storage warning: 256 tokens × 384 × fp16 ≈ 196 KB/frame → ~16 GB for 80k frames. Options: cache
  at fp16 on the GPU machine, or pool to an 8×8 token grid (4× smaller) as a first step.
- New `dino_dynamics.py`: ViT predictor over (H = 3 frames × N tokens), action/proprio embedded and
  concatenated per token, causal over time, teacher-forced. Start small (e.g. 4 layers, 8 heads) — a
  pendulum doesn't need the paper's size.
- Margin / actor / critic: pool tokens → MLP (only the input layer changes vs 1A).
- Every place that does `z_window.flatten(1)` assumes a vector latent — route it through a shared
  `pool(z)` helper instead.
- The filter stays model-free (scores actions with Q) so its cost doesn't grow with the token count.

Exit gate (both 1A and 1B): DINO-based V matches ground truth at least as well as the CNN. If 1A already does, 1B can
wait until the arm, where scenes are complicated enough to need spatial tokens.

---

## Phase 2 — Probabilistic layer, in simulation

Goal: the novel contribution, developed where it can be checked against ground truth.
- Stochastic dynamics: ensemble of K predictors (simplest) or a mean + variance head.
- Particle estimate of `P(V(z') ≥ α V(z))`, certified with Hoeffding / scenario sample counts
  (Mestres et al. 2025, see CLAUDE.md [5]); finite-horizon claim `P(safe for H) ≥ (1 − δ)^H`.
- Validate: inject process noise into the sim pendulum, Monte Carlo the true safety rate, compare to the bound.
- Discuss with Prof. Cosner: how to treat world-model error as the "disturbance" in the bound.

---

## Phase 3 — Physical pendulum

**Open questions to settle first** (hardware isn't specified yet):
- Motor + driver: torque (current) control available? Torque limit? Control rate?
- Joint encoder? (Strongly wanted — see "labels" below.)
- Camera: fps, latency, placement; fixed mount and controlled background/lighting.
- Compute next to the rig: GPU for DINO at the control rate?
- Physical failure: |θ| > θ_fail must be **harmless** (soft stop / padding, low mass) since data
  collection must include failures.

**Plan**
- Rate: control period `dt` = camera period (e.g. 30 Hz → dt 0.033). Rescale horizon and γ (rule 2).
  Torque limit chosen so the viable set is strictly smaller than the constraint set (as in sim).
- Labels and evaluation from the joint encoder: `safe = |θ_enc| ≤ θ_fail` per frame — no hand labeling.
  The encoder also allows an approximate ground truth: fit a pendulum model to logged (θ, ω, u), run
  `ground_truth.py` on it, and compare V to that.
- Observations: **vision only** for the policy and V, to keep the replication faithful. Joint angles
  are used only for labels and evaluation. (Feeding θ into the latent would make the problem trivial.)
- Data: same policy mix as sim (pd, noisy_pd, random, pd_then_random) with wide initial ω,
  started by a scripted kick. Many short episodes; log images, actions, θ_enc, timestamps.
- Train from real data (don't transfer sim weights): cache DINO features → dynamics → margin → value.
- Deployment ladder:
  1. Monitor only — filter computes but doesn't act; log V vs the true outcome.
  2. Filter on, low torque limit, hand on e-stop.
  3. Full evaluation: no filter vs least-restrictive vs LatentCBF (safety rate, |Δa|, task success).
- Latency budget: camera → DINO → Q-sampling → command must fit in `dt`. Measure it; DINO ViT-S+ in fp16
  on GPU is the main cost.

---

## Phase 4 — Unitree D1 arm + hourglass

**Open questions**
- D1 interface (Unitree SDK), action space (joint vs end-effector deltas), control rate, DOF.
- Nominal policy: scripted flipping, teleoperation, or a learned diffusion policy (as in the paper)?
- How to label "sand ran out": the paper hand-labeled the first failure frame (~2.5 h / 1,300 trajectories).
  Alternatives: a timer since the last flip, or a vision-language model labeling the sand level.

**Plan**
- Two cameras (scene + wrist) + joint state, DINO-WM latent exactly as in the paper (Phase 1B code).
- Action chunks (the paper used 16-step chunks from the diffusion policy); data mixing 50/50.
- Safety layers below the filter: joint/velocity limits, workspace bounds, e-stop.
- This failure (sand running out) is physically harmless. The arm's own motion is the hazard, so
  the base limits matter more than the learned filter here.
- Same deployment ladder as Phase 3.

---

## Milestones at a glance

| # | Milestone | Validated against |
|---|---|---|
| 0 | CNN value + LatentCBF filter working in sim | exact ground truth |
| 1A | Pooled-DINO pipeline ≥ CNN in sim | exact ground truth |
| 1B | Token DINO-WM pipeline in sim | exact ground truth |
| 2 | Probabilistic filter + bound vs Monte Carlo | exact ground truth + noise |
| 3 | Physical pendulum: monitor → filter → eval | encoder labels / fitted model |
| 4 | D1 + hourglass | outcome labels |
| — | ICRA submission (September deadline) | — |
