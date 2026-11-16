"""Ground-truth safe set for the pendulum, computed on the TRUE state (theta, omega).

The learned value function only sees latents, but in simulation we know the real state,
so we can solve the safety problem exactly on a grid and use it as an answer key.

Undiscounted safety value iteration (viability kernel):
    V_0(x)     = l(x)
    V_{k+1}(x) = min( l(x), max_a V_k(f(x, a)) )
with l(x) = FAIL_ANGLE - |theta|. V(x) >= 0 means some action sequence keeps the pole
inside |theta| <= FAIL_ANGLE forever; V(x) < 0 means it is doomed, even if l(x) > 0.
"""
import numpy as np
from enviornment import Environment, FAIL_ANGLE


def _physics(env):
    return env.gravity, env.mass, env.pole_length, env.delta_time, env.damping


def step_batch(theta, omega, u, env):
    """Vectorized copy of Environment.step (same constants, same semi-implicit Euler)."""
    g, m, l, dt, c = _physics(env)
    inertia = m * l ** 2
    acc = (g / l) * np.sin(theta) + u / inertia - (c / inertia) * omega
    omega_next = omega + acc * dt
    theta_next = theta + omega_next * dt
    return theta_next, omega_next


def _bilinear(V, thetas, omegas, th, om):
    """Bilinear interpolation of V on the (thetas, omegas) grid; queries are clipped to the grid."""
    th = np.clip(th, thetas[0], thetas[-1])
    om = np.clip(om, omegas[0], omegas[-1])
    ft = (th - thetas[0]) / (thetas[1] - thetas[0])
    fo = (om - omegas[0]) / (omegas[1] - omegas[0])
    i0 = np.clip(np.floor(ft).astype(int), 0, len(thetas) - 2)
    j0 = np.clip(np.floor(fo).astype(int), 0, len(omegas) - 2)
    wt, wo = ft - i0, fo - j0
    return ((1 - wt) * (1 - wo) * V[i0, j0] + wt * (1 - wo) * V[i0 + 1, j0]
            + (1 - wt) * wo * V[i0, j0 + 1] + wt * wo * V[i0 + 1, j0 + 1])


def viability_grid(n_theta=121, n_omega=121, theta_max=0.6, omega_max=3.0,
                   n_actions=21, max_iters=2000, tol=1e-7):
    """Returns (thetas, omegas, V) with V of shape (n_theta, n_omega), indexed V[i_theta, j_omega]."""
    env = Environment()
    a_max = float(env.action_space.high[0])
    thetas = np.linspace(-theta_max, theta_max, n_theta)
    omegas = np.linspace(-omega_max, omega_max, n_omega)
    TH, OM = np.meshgrid(thetas, omegas, indexing="ij")
    actions = np.linspace(-a_max, a_max, n_actions)

    # successor states for every (grid point, action): shape (n_actions, n_theta, n_omega)
    nxt = [step_batch(TH, OM, a, env) for a in actions]
    th_next = np.stack([t for t, _ in nxt])
    om_next = np.stack([o for _, o in nxt])

    margin = FAIL_ANGLE - np.abs(TH)
    V = margin.copy()
    for _ in range(max_iters):
        best_next = _bilinear(V, thetas, omegas, th_next, om_next).max(axis=0)
        V_new = np.minimum(margin, best_next)
        if np.max(np.abs(V_new - V)) < tol:
            V = V_new
            break
        V = V_new
    return thetas, omegas, V


if __name__ == "__main__":
    thetas, omegas, V = viability_grid()
    j0 = np.argmin(np.abs(omegas))  # omega = 0 column
    safe_at_rest = thetas[V[:, j0] >= 0]
    print(f"At omega = 0, viable for theta in [{safe_at_rest.min():+.3f}, {safe_at_rest.max():+.3f}] "
          f"(failure set starts at +-{FAIL_ANGLE})")
    print(f"Viable fraction of the grid inside |theta| <= {FAIL_ANGLE}: "
          f"{np.mean(V[np.abs(thetas) <= FAIL_ANGLE] >= 0):.3f}")
