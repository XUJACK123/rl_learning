"""Iterative LQG (iLQG / DDP) planner over the MuJoCo humanoid.

Follows the shooting-based iLQG used by MJPC's gradient planner (Tassa et al.):
  - dynamics linearised per timestep with mujoco.mjd_transitionFD;
  - cost quadratics from a Gauss-Newton approximation of the residual reward;
  - backward Riccati pass with Levenberg-Marquardt regularisation;
  - forward line search on the rollout cost.

Pure NumPy + MuJoCo; no other dependencies. The planner runs on its own mjData
so the caller's data is never clobbered mid-rollout.
"""

from __future__ import annotations

import numpy as np
import mujoco

from env import HumanoidEnv, StandParams


class iLQG:
    def __init__(
        self,
        env: HumanoidEnv,
        params: StandParams,
        horizon: int = 40,
        max_iters: int = 5,
        mu_min: float = 1e-6,
        mu_max: float = 1e8,
        dmu: float = 1.5,
        dmu0: float = 1.1,
        cost_tol: float = 1e-4,
        alphas: tuple[float, ...] = (1.0, 0.5, 0.25, 0.125, 0.0625),
    ):
        self.env = env
        self.params = params
        self.T = horizon
        self.max_iters = max_iters
        self.mu_min, self.mu_max = mu_min, mu_max
        self.dmu, self.dmu0 = dmu, dmu0
        self.cost_tol = cost_tol
        self.alphas = alphas

        self.d = mujoco.MjData(env.m)
        self.d_tmp = mujoco.MjData(env.m)
        self.nx = env.nq + env.nv
        self.nu = env.nu
        self.eps_dyn = 1e-6
        self.u_lo = env.m.actuator_ctrlrange[:, 0].copy()
        self.u_hi = env.m.actuator_ctrlrange[:, 1].copy()

        self.K: list[np.ndarray] = []
        self.k: list[np.ndarray] = []
        self.x_hat: list[np.ndarray] = []
        self.last_cost = np.inf
        self.t0 = 0.0   # 本次规划的起始时间（秒）

    # ---------------------------------------------------------------- helpers
    def _set(self, x: np.ndarray, u: np.ndarray) -> None:
        self.env.set_state(x, self.d)
        self.d.ctrl[:] = u

    def _step(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        self._set(x, u)
        mujoco.mj_step(self.env.m, self.d)
        return self.env.get_state(self.d)

    def _cost_at(self, x: np.ndarray, u: np.ndarray, t: int) -> float:
        self._set(x, u)
        return self.env.cost(self.params, self.d, self.t0 + t * self.env.dt)

    def _dynamics_jacobians(self, x: np.ndarray, u: np.ndarray):
        # Finite-difference dynamics linearisation in the full 55-dim state
        # (mjd_transitionFD works in reduced coordinates for free-joint systems,
        # so a self-contained full-space FD keeps the algebra unambiguous).
        A = np.zeros((self.nx, self.nx))
        B = np.zeros((self.nx, self.nu))
        self._set(x, u)
        mujoco.mj_step(self.env.m, self.d)
        x_next = self.env.get_state(self.d)

        for j in range(self.nx):
            xp = x.copy()
            xp[j] += self.eps_dyn
            self.env.set_state(xp, self.d_tmp)
            self.d_tmp.ctrl[:] = u
            mujoco.mj_step(self.env.m, self.d_tmp)
            A[:, j] = (self.env.get_state(self.d_tmp) - x_next) / self.eps_dyn

        for j in range(self.nu):
            up = u.copy()
            up[j] += self.eps_dyn
            self._set(x, up)
            mujoco.mj_step(self.env.m, self.d)
            B[:, j] = (self.env.get_state(self.d) - x_next) / self.eps_dyn

        self._set(x, u)  # restore nominal workspace
        return A, B

    def _cost_quadratics(self, x: np.ndarray, u: np.ndarray, t: int):
        return self.env.cost_quadratics(
            self.params, x, u, self.d, time=self.t0 + t * self.env.dt)

    # -------------------------------------------------------------- rollouts
    def forward(self, x0: np.ndarray, U: np.ndarray, alpha: float,
                apply_feedback: bool = True) -> tuple:
        xs = [x0.copy()]
        us: list[np.ndarray] = []
        cost = 0.0
        x = x0
        for t in range(self.T):
            u = U[t]
            if apply_feedback:
                u = u + alpha * self.k[t] + self.K[t] @ (x - self.x_hat[t])
            u = np.clip(u, self.u_lo, self.u_hi)
            us.append(u)
            cost += self._cost_at(x, u, t)
            x = self._step(x, u)
            xs.append(x)
        cost += self._cost_at(x, np.zeros(self.nu), self.T)  # terminal state cost
        return xs, np.asarray(us), cost

    # ----------------------------------------------------------- backward pass
    def backward(self, mu: float) -> None:
        k, K = [], []

        # terminal cost quadratics (no control at the terminal state)
        lx, _, lxx, _, _ = self._cost_quadratics(
            self.x_hat[-1], np.zeros(self.nu), self.T)
        Vx, Vxx = lx, lxx

        for t in reversed(range(self.T)):
            A, B = self._dynamics_jacobians(self.x_hat[t], self.u_hat[t])
            lx, lu, lxx, luu, lux = self._cost_quadratics(
                self.x_hat[t], self.u_hat[t], t)

            Qx = lx + A.T @ Vx
            Qu = lu + B.T @ Vx
            Qxx = lxx + A.T @ Vxx @ A
            Quu = luu + B.T @ Vxx @ B
            Qux = lux + B.T @ Vxx @ A

            # Levenberg-Marquardt regularisation on the control Hessian
            Quu_reg = Quu + mu * np.eye(self.nu)
            try:
                kt = -np.linalg.solve(Quu_reg, Qu)
                Kt = -np.linalg.solve(Quu_reg, Qux)
            except np.linalg.LinAlgError:
                raise

            Vx = Qx + Qux.T @ kt
            Vxx = Qxx + Qux.T @ Kt
            k.append(kt)
            K.append(Kt)

        self.k = list(reversed(k))
        self.K = list(reversed(K))

    # ------------------------------------------------------------------- plan
    def plan(self, x0: np.ndarray, U: np.ndarray | None = None,
             shift: int = 1) -> np.ndarray:
        """Return an optimised control sequence U (T x nu) from state x0."""
        # 记录本次规划的起始时间（用于步态相位等时变 reward 项）
        self.t0 = self.env.time
        if U is None or U.shape != (self.T, self.nu):
            U = np.zeros((self.T, self.nu))
        else:
            # warm start: shift the previous plan by the elapsed steps
            shift = min(shift, self.T)
            U = np.vstack([U[shift:], np.repeat(U[-1:], shift, axis=0)])

        # initial nominal trajectory with feedback disabled
        self.k = [np.zeros(self.nu) for _ in range(self.T)]
        self.K = [np.zeros((self.nu, self.nx)) for _ in range(self.T)]
        self.x_hat, self.u_hat, J_prev = self.forward(x0, U, 1.0, False)

        mu = self.mu_min
        for _ in range(self.max_iters):
            try:
                self.backward(mu)
            except np.linalg.LinAlgError:
                mu = min(mu * self.dmu, self.mu_max)
                if mu >= self.mu_max:
                    break
                continue

            improved = False
            for alpha in self.alphas:
                xs, us, J_new = self.forward(x0, U, alpha)
                if J_new < J_prev - self.cost_tol * abs(J_prev):
                    self.x_hat, self.u_hat = xs, us
                    U = us
                    J_prev = J_new
                    mu = max(mu / self.dmu0, self.mu_min)
                    improved = True
                    break

            if not improved:
                mu = min(mu * self.dmu, self.mu_max)
                if mu >= self.mu_max:
                    break

        self.last_cost = J_prev
        return U
