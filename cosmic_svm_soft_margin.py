import os
import time

import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

plt.rcParams.update({
    "font.family": "serif",          # match the journal body text
    "font.size": 15,
    "axes.labelsize": 15,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 15,
    "mathtext.fontset": "cm",        # Computer Modern math, matches LaTeX
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.2,
})
# =============================================================================
# Settings
# =============================================================================
# --- SVM formulation ----------------------------------------------------------
#   "hard": min 0.5||w||^2                         s.t. 1 - y_i(w.x_i + b) <= 0
#   "l1"  : min 0.5||w||^2 + C*sum(xi)             s.t. 1 - y_i(w.x_i + b) - xi_i <= 0,  -xi_i <= 0
#   "l2"  : min 0.5||w||^2 + (C/2)*||xi||^2        s.t. 1 - y_i(w.x_i + b) - xi_i <= 0
#           (xi >= 0 is implied at the optimum, so it is not imposed)
SVM_MODE = "l2"
C_SVM = 1.0
POS_CLASS = 2         # label +1 for this Iris class, -1 for the other two (all 150 samples used)
                      # hard margin needs POS_CLASS = 0 (setosa): the only linearly separable class

FOLDER = "/home/rawan/HSGD/CoO/COSMIC/results"
TAG = f"SECOND_SOFT_PEN_SVM_Iris_{SVM_MODE}" + ("" if SVM_MODE == "hard" else f"_C{C_SVM:g}") + "_COSMO_FL_KKTSMC_Adam"
N_ITER = 20000        # iterations per run
N_RUNS = 10           # random initial points, shared by all methods
KKT_TOL = 1e-14        # stop when KKT residual < KKT_TOL (None = always N_ITER)
FL_STEP_TOL = 1e-14    # FL stops when ||x_new - x|| < FL_STEP_TOL (as in SQP_ineq); 0 disables
ERR_TOL = 1e-3        # error level for "CPU time to reach tolerance"
SAVE_EVERY = 10       # thinning of the per-iteration CSV (final iterate always kept)
GRID_POINTS = 400     # resolution of the error-vs-CPU-time table
CPU_XMAX = None       # right limit of the error-vs-CPU-time plot (None = automatic)
SHOW_PLOTS = True

# COSMO parameters
#   DT = 0.004 (old value) diverged on the 150-sample problems in my NumPy replica;
#   DT = 0.0025 with K = 100..200 converged for hard / l1 / l2. K <= 20 does not converge.
DT = 0.0025
BETA = 0.1
EPS_VAL = 1e-10
EPS_SAT = 1e-2
R_EPS, EPS_MIN, EPS_MAX, C_SHRINK = 1.05, 1e-16, 1e-1, 0.85
K_COSMO = [50,100, 150]

# FL parameters (SQP_ineq, no Hessian, no momentum)
ETA_FL = 0.1
K_FL = [5]
FL_REUSE_MODEL = True  # build the Gurobi lambda-QP once (J J^T is constant) and only swap the linear term

# KKT-SMC parameters
ETA_SMC, Q_X, EPS_X, PHI_X, RHO_C = 0.8, 1.0, 1e-3, 0.01, 1e-15

# Adam (shared by the penalty and augmented-Lagrangian variants)
ADAM_B1, ADAM_B2, ADAM_EPS = 0.9, 0.999, 1e-8
ADAM_LR_DECAY = 1e-2   # lr_k = lr * ADAM_LR_DECAY**(k / N_ITER); 1.0 = constant step size

# Adam on a quadratic penalty with a scheduled penalty parameter:
#   F_k(x) = f(x) + rho_k/2 * ||max(0, g(x))||^2,   rho_k = rho_start * (rho_end/rho_start)**(k/N_ITER)
#   rho_end > rho_start : rho grows geometrically (penalty weight 1/rho decays), so the O(1/rho) bias shrinks
#   rho_end = rho_start : constant rho (previous behaviour)
#   rho_end < rho_start : rho shrinks (the bias then grows towards 1/rho_end)
#   rho_end >~ 1e5 is too stiff for Adam in my tests.
PEN_CONFIGS = [dict(rho_start=10.0, rho_end=1e4, lr=0.03),
               dict(rho_start=10.0, rho_end=1e4, lr=0.1)]

# Adam inside an augmented Lagrangian (PHR form for g(x) <= 0):
#   L(x, mu) = f(x) + 1/(2 rho) * sum( max(0, mu + rho g(x))^2 - mu^2 )
#   `inner` Adam steps on x for fixed mu, then mu <- max(0, mu + rho g(x)); N_ITER counts Adam steps
AL_CONFIGS = [dict(rho=10.0, lr=0.1, inner=50),
              dict(rho=10.0, lr=0.03, inner=50)]

os.makedirs(FOLDER, exist_ok=True)

# =============================================================================
# Problem: linear SVM on the full Iris set (all 150 samples), x = [w, b, (xi)]
# =============================================================================
if SVM_MODE not in ("hard", "l1", "l2"):
    raise ValueError("SVM_MODE must be 'hard', 'l1' or 'l2'")
if SVM_MODE == "hard" and POS_CLASS != 0:
    raise ValueError("Hard margin is only feasible for setosa vs rest (POS_CLASS = 0)")

iris = load_iris()
X = StandardScaler().fit_transform(iris.data)
y = np.where(iris.target == POS_CLASS, 1.0, -1.0)
n_samples, n_features = X.shape
n_vars = n_features + 1 + (0 if SVM_MODE == "hard" else n_samples)   # w, b, slacks
m = {"hard": n_samples, "l1": 2 * n_samples, "l2": n_samples}[SVM_MODE]   # number of constraints
X_j = jnp.array(X)
y_j = jnp.array(y)
print(f"SVM_MODE={SVM_MODE}, C={C_SVM}, n_vars={n_vars}, m={m}")


def objective(x):
    w = x[:n_features]
    f = 0.5 * jnp.dot(w, w)
    if SVM_MODE == "l1":
        f = f + C_SVM * jnp.sum(x[n_features + 1:])
    elif SVM_MODE == "l2":
        xi = x[n_features + 1:]
        f = f + 0.5 * C_SVM * jnp.dot(xi, xi)
    return f


def ineq_constraint(x):               # g(x) <= 0
    w, b = x[:n_features], x[n_features]
    margin = 1.0 - y_j * (X_j @ w + b)
    if SVM_MODE == "hard":
        return margin
    xi = x[n_features + 1:]
    g = margin - xi
    if SVM_MODE == "l1":
        g = jnp.concatenate([g, -xi])
    return g


# =============================================================================
# JIT-compiled problem functions (built once, reused by every run)
# =============================================================================
def build_problem(objective, ineq):
    g_fun = jax.jit(lambda x: jnp.atleast_1d(ineq(x)))
    g_jac = jax.jit(jax.jacobian(lambda x: jnp.atleast_1d(ineq(x))))
    f_grad = jax.jit(jax.grad(objective))
    lag_hess = jax.jit(lambda x, lam: jax.hessian(
        lambda z: objective(z) + lam @ jnp.atleast_1d(ineq(z)))(x))

    @jax.jit
    def kkt(x, lam):
        """max(stationarity, primal infeas, complementarity, dual infeas)."""
        g = jnp.atleast_1d(ineq(x))
        J = jax.jacobian(lambda z: jnp.atleast_1d(ineq(z)))(x)
        stat = jnp.linalg.norm(jax.grad(objective)(x) + J.T @ lam)
        feas = jnp.maximum(jnp.max(g), 0.0)
        comp = jnp.max(jnp.abs(lam * g))
        dual = jnp.maximum(jnp.max(-lam), 0.0)
        return jnp.max(jnp.stack([stat, feas, comp, dual]))

    return dict(g=g_fun, J=g_jac, grad=f_grad, hess=lag_hess, kkt=kkt)


P = build_problem(objective, ineq_constraint)


# =============================================================================
# Reference solution: Gurobi QP, then an active-set polish to ~1e-14
#   f is quadratic and g is affine in every SVM_MODE:
#       f(x) = 0.5 x'Hx + c'x,   g(x) = Jx + d
#   (SLSQP is not used: it stops with a line-search failure on the l1 problem.)
# =============================================================================
x_zero = jnp.zeros(n_vars)
H_qp = np.asarray(jax.hessian(objective)(x_zero))
c_qp = np.asarray(jax.grad(objective)(x_zero))
J_qp = np.asarray(jax.jacobian(ineq_constraint)(x_zero))
d_qp = np.asarray(ineq_constraint(x_zero))
x_chk = np.random.default_rng(0).standard_normal(n_vars)        # guard: f quadratic, g affine
assert np.isclose(0.5 * x_chk @ H_qp @ x_chk + c_qp @ x_chk, float(objective(jnp.asarray(x_chk))))
assert np.allclose(J_qp @ x_chk + d_qp, np.asarray(ineq_constraint(jnp.asarray(x_chk))))


def solve_qp_gurobi(H, c, J, d):
    mdl = gp.Model()
    mdl.setParam('OutputFlag', 0)
    mdl.setParam('Method', 2)              # barrier
    mdl.setParam('BarConvTol', 1e-10)
    xv = mdl.addMVar(shape=H.shape[0], lb=-GRB.INFINITY, name="x")
    mdl.setObjective(0.5 * xv @ H @ xv + c @ xv, GRB.MINIMIZE)
    mdl.addConstr(J @ xv <= -d)
    mdl.optimize()
    if mdl.status != GRB.OPTIMAL:
        raise RuntimeError(f"Reference QP not solved to optimality (status {mdl.status})")
    return xv.X


def polish_active_set(x, H, c, J, d, act_tol=1e-6, max_it=200):
    """Working-set polish. Start from the near-active constraints, solve the equality-constrained
    KKT system, add the most violated constraint / drop the most negative multiplier, repeat."""
    n, m_ = H.shape[0], J.shape[0]
    W = set(np.where(J @ x + d > -act_tol)[0].tolist())
    for _ in range(max_it):
        idx = np.array(sorted(W), dtype=int)
        Ja = J[idx]
        Kmat = np.block([[H, Ja.T], [Ja, np.zeros((len(idx), len(idx)))]])
        sol = np.linalg.lstsq(Kmat, np.concatenate([-c, -d[idx]]), rcond=None)[0]
        x, la = sol[:n], sol[n:]
        gv = J @ x + d
        outside = np.ones(m_, dtype=bool)
        outside[idx] = False
        viol = np.where(outside & (gv > 1e-12))[0]
        if viol.size:
            W.add(int(viol[np.argmax(gv[viol])]))
            continue
        if la.size and la.min() < -1e-12:
            W.discard(int(idx[np.argmin(la)]))
            continue
        break
    lam = np.zeros(m_)
    lam[idx] = la
    return x, lam


x_qp = solve_qp_gurobi(H_qp, c_qp, J_qp, d_qp)
x_opt, lam_opt = polish_active_set(x_qp, H_qp, c_qp, J_qp, d_qp)
kkt_ref = float(P["kkt"](jnp.asarray(x_opt), jnp.asarray(lam_opt)))
print(f"Reference: f*={float(objective(jnp.asarray(x_opt))):.10f}, "
      f"|x_opt - x_gurobi|={np.linalg.norm(x_opt - x_qp):.2e}, KKT residual={kkt_ref:.2e}")
print(f"  active constraints: {int(np.sum(lam_opt > 1e-9))}, "
      f"w={x_opt[:n_features]}, b={x_opt[n_features]:.6f}")
if SVM_MODE != "hard":
    print(f"  samples with slack > 0: {int(np.sum(x_opt[n_features + 1:] > 1e-9))} / {n_samples}")
if kkt_ref > 1e-9:
    print("WARNING: reference KKT residual > 1e-9 -- tighten the Gurobi tolerances or increase act_tol.")

if SVM_MODE == "l1":
    svc = SVC(kernel='linear', C=C_SVM, tol=1e-10).fit(X, y)
    print(f"sklearn SVC check (w,b): |x_SVC - x_opt| = "
          f"{np.linalg.norm(np.append(svc.coef_.ravel(), svc.intercept_) - x_opt[:n_features + 1]):.2e}")
elif SVM_MODE == "hard":
    svc = SVC(kernel='linear', C=1e10).fit(X, y)
    print(f"sklearn SVC check (w,b): |x_SVC - x_opt| = "
          f"{np.linalg.norm(np.append(svc.coef_.ravel(), svc.intercept_) - x_opt):.2e}")


# =============================================================================
# COSMO
# =============================================================================
def phi_fn(g, xi, eps):
    return xi - g - jnp.sqrt(g**2 + xi**2 + eps**2)


def sat(v, eps_sat):
    return jnp.clip(v / eps_sat, -1.0, 1.0)


def eps_update(eps_vec, phi_val, mode, r_eps, eps_min, eps_max, c_shrink):
    if mode == "decay":
        shrink_mask = jnp.abs(phi_val) < c_shrink * eps_vec
        eps_shrunk = jnp.maximum(eps_vec / r_eps, eps_min)
        return jnp.where(shrink_mask, eps_shrunk, eps_vec)
    elif mode == "grow":
        return jnp.minimum(eps_vec * r_eps, eps_max)
    else:
        return eps_vec


def make_cosmo_step(K):
    """One COSMO iteration (state update + eps update), jitted once per K.
    M and G are diagonal, so they are applied elementwise (no m x m matrices)."""
    delta_M = DT * K / 2.0

    @jax.jit
    def step(x, xi, eps_vec):
        g_x = jnp.atleast_1d(ineq_constraint(x))
        J_x = jax.jacobian(lambda z: jnp.atleast_1d(ineq_constraint(z)))(x)
        grad_f = jax.grad(objective)(x)

        x_dot = -grad_f - J_x.T @ xi
        x_next = x + DT * x_dot

        phi_val = phi_fn(g_x, xi, eps_vec)
        s = jnp.sqrt(g_x**2 + xi**2 + eps_vec**2)
        d = 1.0 - xi / s
        d_tilde = jnp.sqrt(d**2 + delta_M**2)

        xi_dot = (-K * phi_val - BETA * sat(phi_val, EPS_SAT)
                  + (1.0 + g_x / s) * (J_x @ x_dot)) / d_tilde
        xi_next = xi + DT * xi_dot
        eps_next = eps_update(eps_vec, phi_val, "decay", R_EPS, EPS_MIN, EPS_MAX, C_SHRINK)
        return x_next, xi_next, eps_next

    return step


# =============================================================================
# Feedback linearisation (FL), from SQP_ineq
# =============================================================================
def solve_lambda(J_h, grad_f, h_x, K):
    """Copied from solve_lambda (non-Hessian branch). Threads=1 is added so the
    CPU-time measurement is not spread across Gurobi's worker threads."""
    m_ = J_h.shape[0]
    model = gp.Model()
    model.setParam('OutputFlag', 0)
    model.setParam('Threads', 1)

    lambda_ = model.addMVar(shape=m_, name="lambda", lb=0.0)

    J_h_J_hT = J_h @ J_h.T
    J_h_grad_f = J_h @ grad_f
    K_h_x = K @ h_x

    model.setObjective(0.5 * lambda_ @ J_h_J_hT @ lambda_
                       + (J_h_grad_f - K_h_x) @ lambda_, GRB.MINIMIZE)
    model.optimize()
    if not model.status == GRB.OPTIMAL:
        print('flag')
    return lambda_.X


class FLSolver:
    """Same lambda-QP as solve_lambda, but the Hessian J J^T is built once (J is constant for every
    SVM_MODE because the constraints are affine); each call only swaps the linear term."""

    def __init__(self, J):
        self.model = gp.Model()
        self.model.setParam('OutputFlag', 0)
        self.model.setParam('Threads', 1)
        self.lam = self.model.addMVar(shape=J.shape[0], name="lambda", lb=0.0)
        self.model.setObjective(0.5 * self.lam @ (J @ J.T) @ self.lam, GRB.MINIMIZE)
        self.model.update()

    def solve(self, q):
        self.lam.setAttr("Obj", q)
        self.model.optimize()
        if not self.model.status == GRB.OPTIMAL:
            print('flag')
        return self.lam.X


def run_fl(x0, m_, eta, K_gain, N, kkt_fn, step_tol):
    """SQP_ineq (non-Hessian) loop. K_gain * I makes K @ h_x valid for a scalar gain."""
    x = np.asarray(x0, dtype=float).copy()
    Kmat = K_gain * np.eye(m_)
    fl_solver = FLSolver(np.asarray(P["J"](x))) if FL_REUSE_MODEL else None
    log = new_log()
    for k in range(N + 1):
        t0 = time.process_time()
        grad_f = np.asarray(P["grad"](x))
        h_x = np.asarray(P["g"](x))
        J_h = np.asarray(P["J"](x))
        if fl_solver is None:
            lam = solve_lambda(J_h, grad_f, h_x, Kmat)
        else:
            lam = fl_solver.solve(J_h @ grad_f - K_gain * h_x)
        x_next = x - eta * (grad_f + J_h.T @ lam) if k < N else x
        dt_k = time.process_time() - t0

        log_state(log, x, lam, kkt_fn(x, lam))
        if k == N or (KKT_TOL is not None and log["kkt"][-1] < KKT_TOL):
            break
        if step_tol > 0 and np.linalg.norm(x_next - x) < step_tol:
            break
        log["cpu_step"].append(dt_k)
        x = x_next
    return log


# =============================================================================
# KKT residual sliding-manifold solver (SM-KKT)  -- unchanged
# =============================================================================
def new_log():
    return {"x": [], "lam": [], "kkt": [], "cpu_step": []}


def log_state(log, x, lam, kkt_val):
    log["x"].append(np.array(x, dtype=float))
    log["lam"].append(np.array(lam, dtype=float))
    log["kkt"].append(float(kkt_val))


def run_cosmo(step, x0, m_, epsilon, N, kkt_fn):
    x = jnp.asarray(x0)
    xi = jnp.zeros(m_)
    eps_vec = jnp.ones(m_) * epsilon
    log = new_log()
    log_state(log, x, xi, kkt_fn(x, xi))
    for k in range(N):
        t0 = time.process_time()
        x, xi, eps_vec = step(x, xi, eps_vec)
        jax.block_until_ready((x, xi, eps_vec))
        log["cpu_step"].append(time.process_time() - t0)
        log_state(log, x, xi, kkt_fn(x, xi))
        if KKT_TOL is not None and log["kkt"][-1] < KKT_TOL:
            break
    return log


def run_kktsmc(P_, x0, m_, kkt_fn, eta, q, eps, phi, rho, N, lam0=0.1):
    x = np.asarray(x0, dtype=float).copy()
    lam = np.full(m_, lam0)
    log = new_log()
    for k in range(N + 1):
        t0 = time.process_time()
        if k < N:
            gf = np.asarray(P_["grad"](x))
            g = np.asarray(P_["g"](x))
            J = np.asarray(P_["J"](x))
            H = np.asarray(P_["hess"](x, lam))
            s = np.sqrt(lam**2 + g**2 + rho)
            Sx = gf + J.T @ lam
            Sc = -lam + g + s
            Dg = 1.0 + g / s
            Dl = -1.0 + lam / s
            rx = -(q * Sx + eps * np.clip(Sx / phi, -1.0, 1.0))
            rc = -(q * Sc + eps * np.clip(Sc / phi, -1.0, 1.0))
            Hbar = H - J.T @ np.diag(Dg / Dl) @ J
            rbar = rx - J.T @ (rc / Dl)
            try:
                dx = np.linalg.solve(Hbar, rbar)
            except np.linalg.LinAlgError:
                dx = np.linalg.lstsq(Hbar, rbar, rcond=None)[0]
            dl = (rc - Dg * (J @ dx)) / Dl
            x_next = x + eta * dx
            lam_next = lam + eta * dl
        dt_k = time.process_time() - t0
        log_state(log, x, lam, kkt_fn(x, lam))
        if k == N or (KKT_TOL is not None and log["kkt"][-1] < KKT_TOL):
            break
        log["cpu_step"].append(dt_k)
        x, lam = x_next, lam_next
    return log


# =============================================================================
# Adam on a penalty function, and Adam inside an augmented Lagrangian
# =============================================================================
def penalty_loss(x, rho):
    """Quadratic penalty  f(x) + rho/2 * ||max(0, g(x))||^2."""
    g = jnp.atleast_1d(ineq_constraint(x))
    return objective(x) + 0.5 * rho * jnp.sum(jnp.maximum(g, 0.0) ** 2)


def al_loss(x, mu, rho):
    """Augmented Lagrangian for g(x) <= 0 (multiplier mu >= 0, penalty rho)."""
    g = jnp.atleast_1d(ineq_constraint(x))
    return objective(x) + (jnp.sum(jnp.maximum(mu + rho * g, 0.0) ** 2)
                           - jnp.sum(mu ** 2)) / (2.0 * rho)


def make_adam_step(loss_fn):
    """One Adam step on loss_fn(x, *aux), jitted once. aux = (rho,) or (mu, rho)."""
    grad_fn = jax.grad(loss_fn)

    @jax.jit
    def step(x, m1, m2, t, lr, *aux):
        gr = grad_fn(x, *aux)
        m1 = ADAM_B1 * m1 + (1.0 - ADAM_B1) * gr
        m2 = ADAM_B2 * m2 + (1.0 - ADAM_B2) * gr**2
        m1_hat = m1 / (1.0 - ADAM_B1**t)
        m2_hat = m2 / (1.0 - ADAM_B2**t)
        return x - lr * m1_hat / (jnp.sqrt(m2_hat) + ADAM_EPS), m1, m2

    return step


adam_pen_step = make_adam_step(penalty_loss)
adam_al_step = make_adam_step(al_loss)


@jax.jit
def lam_penalty(x, rho):
    """Multiplier implied by the penalty: lam = rho * max(0, g(x)) (used for the KKT residual)."""
    return rho * jnp.maximum(jnp.atleast_1d(ineq_constraint(x)), 0.0)


@jax.jit
def al_multiplier(x, mu, rho):
    """Multiplier update mu+ = max(0, mu + rho g(x)); also the multiplier used for the KKT residual."""
    return jnp.maximum(mu + rho * jnp.atleast_1d(ineq_constraint(x)), 0.0)


def adam_lr(lr, k, N):
    return float(lr * ADAM_LR_DECAY ** (k / N))


def penalty_rho(rho_start, rho_end, k, N):
    """Geometric schedule from rho_start (k = 0) towards rho_end (k = N)."""
    return float(rho_start * (rho_end / rho_start) ** (k / N))


def run_adam_pen(x0, rho_start, rho_end, lr, N, kkt_fn):
    x = jnp.asarray(x0)
    m1, m2 = jnp.zeros_like(x), jnp.zeros_like(x)
    lam = lam_penalty(x, penalty_rho(rho_start, rho_end, 0, N))
    log = new_log()
    log_state(log, x, lam, kkt_fn(x, lam))
    for k in range(N):
        lr_k = adam_lr(lr, k, N)
        rho_k = penalty_rho(rho_start, rho_end, k, N)
        t0 = time.process_time()
        x, m1, m2 = adam_pen_step(x, m1, m2, k + 1.0, lr_k, rho_k)
        jax.block_until_ready(x)
        log["cpu_step"].append(time.process_time() - t0)
        lam = lam_penalty(x, rho_k)        # logging only, outside the timed region
        log_state(log, x, lam, kkt_fn(x, lam))
        if KKT_TOL is not None and log["kkt"][-1] < KKT_TOL:
            break
    return log


def run_adam_al(x0, m_, rho, lr, inner, N, kkt_fn):
    rho = float(rho)
    x = jnp.asarray(x0)
    mu = jnp.zeros(m_)
    m1, m2 = jnp.zeros_like(x), jnp.zeros_like(x)
    lam = al_multiplier(x, mu, rho)
    log = new_log()
    log_state(log, x, lam, kkt_fn(x, lam))
    for k in range(N):
        lr_k = adam_lr(lr, k, N)
        t0 = time.process_time()
        x, m1, m2 = adam_al_step(x, m1, m2, k + 1.0, lr_k, mu, rho)
        if (k + 1) % inner == 0:           # outer (multiplier) update
            mu = al_multiplier(x, mu, rho)
        jax.block_until_ready((x, mu))
        log["cpu_step"].append(time.process_time() - t0)
        lam = al_multiplier(x, mu, rho)    # logging only, outside the timed region
        log_state(log, x, lam, kkt_fn(x, lam))
        if KKT_TOL is not None and log["kkt"][-1] < KKT_TOL:
            break
    return log

def finalize(log, x_star):
    x_arr = np.array(log["x"])
    err = np.linalg.norm(x_arr - x_star, axis=1)      # over all variables (w, b and slacks)
    cpu_cum = np.concatenate([[0.0], np.cumsum(log["cpu_step"])])
    return dict(err=err, cpu_cum=cpu_cum, kkt=np.array(log["kkt"]), iters=len(err) - 1)

# =============================================================================
# Initial points (same for all methods)
# =============================================================================
np.random.seed(0)
x0_list = []
for _ in range(N_RUNS):
    i = np.random.randint(0, 1000000)
    x0_list.append(np.asarray(
        jax.random.uniform(jax.random.PRNGKey(i), shape=(n_vars,), minval=-5, maxval=5)))

# =============================================================================
# Warm-up: compile JAX functions and start the Gurobi environment before timing
# =============================================================================
x_w = np.asarray(x0_list[0], dtype=float)
lam_w = np.zeros(m)
jax.block_until_ready(P["grad"](x_w))
jax.block_until_ready(P["g"](x_w))
jax.block_until_ready(P["J"](x_w))
jax.block_until_ready(P["hess"](x_w, lam_w))
jax.block_until_ready(P["kkt"](x_w, lam_w))
if FL_REUSE_MODEL:
    FLSolver(np.asarray(P["J"](x_w))).solve(np.asarray(P["J"](x_w)) @ np.asarray(P["grad"](x_w))
                                            - K_FL[0] * np.asarray(P["g"](x_w)))
else:
    solve_lambda(np.asarray(P["J"](x_w)), np.asarray(P["grad"](x_w)),
                 np.asarray(P["g"](x_w)), K_FL[0] * np.eye(m))
x_wj = jnp.asarray(x_w)
z_w = jnp.zeros_like(x_wj)
jax.block_until_ready(adam_pen_step(x_wj, z_w, z_w, 1.0, 0.1, 1.0))
jax.block_until_ready(adam_al_step(x_wj, z_w, z_w, 1.0, 0.1, jnp.zeros(m), 1.0))
jax.block_until_ready(lam_penalty(x_wj, 1.0))
jax.block_until_ready(al_multiplier(x_wj, jnp.zeros(m), 1.0))

# =============================================================================
# Experiments
# =============================================================================
configs = [dict(method="COSMIC", params=f"K={K}", K=K) for K in K_COSMO]
configs += [dict(method="FL", params=f"K={K}", K=K) for K in K_FL]
configs += [dict(method="KKT-SMC", params="-", K=None)]
configs += [dict(method="Adam-Pen", params=f"rho={c['rho_start']:g}->{c['rho_end']:g},lr={c['lr']:g}", **c)
            for c in PEN_CONFIGS]
configs += [dict(method="Adam-AL", params=f"rho={c['rho']:g},T={c['inner']},lr={c['lr']:g}", **c)
            for c in AL_CONFIGS]

results = []
for cfg in configs:
    method, params = cfg["method"], cfg["params"]
    if method == "COSMIC":
        step = make_cosmo_step(cfg["K"])
        jax.block_until_ready(step(jnp.asarray(x0_list[0]), jnp.zeros(m),
                                   jnp.ones(m) * EPS_VAL))

    for run, x0 in enumerate(x0_list):
        if method == "COSMIC":
            log = run_cosmo(step, x0, m, EPS_VAL, N_ITER, P["kkt"])
        elif method == "FL":
            log = run_fl(x0, m, ETA_FL, cfg["K"], N_ITER, P["kkt"], FL_STEP_TOL)
        elif method == "KKT-SMC":
            log = run_kktsmc(P, x0, m, P["kkt"], ETA_SMC, Q_X, EPS_X, PHI_X, RHO_C, N_ITER)
        elif method == "Adam-Pen":
            log = run_adam_pen(x0, cfg["rho_start"], cfg["rho_end"], cfg["lr"], N_ITER, P["kkt"])
        else:  # Adam-AL
            log = run_adam_al(x0, m, cfg["rho"], cfg["lr"], cfg["inner"], N_ITER, P["kkt"])

        res = finalize(log, x_opt)
        results.append(dict(method=method, params=params, run=run, **res))
        print(f"{method:8s} {params:22s} run {run:2d}: iters={res['iters']}, "
              f"cpu={res['cpu_cum'][-1]:.3f} s, final err={res['err'][-1]:.2e}")


def runs_of(cfg):
    return [r for r in results if r["method"] == cfg["method"] and r["params"] == cfg["params"]]


def pad_stack(arrs):
    L = max(len(a) for a in arrs)
    out = np.full((len(arrs), L), np.nan)
    for i, a in enumerate(arrs):
        out[i, :len(a)] = a
    return out


# =============================================================================
# Output 1: per-iteration table (iteration index AND CPU time)
# =============================================================================
rows = []
for r in results:
    n = len(r["err"])
    idx = np.unique(np.r_[np.arange(0, n, SAVE_EVERY), n - 1])
    step_s = np.r_[np.diff(r["cpu_cum"]), np.nan]
    rows.append(pd.DataFrame(dict(
        method=r["method"], params=r["params"], run=r["run"],
        iter=idx,
        cpu_cum_s=r["cpu_cum"][idx],
        cpu_step_s=step_s[idx],
        kkt=r["kkt"][idx],
        err=r["err"][idx],
    )))
iterations_df = pd.concat(rows, ignore_index=True)
iterations_df.to_csv(f"{FOLDER}/iterations_{TAG}.csv", index=False)

# =============================================================================
# Output 2: error as a function of CPU time (common time grid per method)
# =============================================================================
grid_frames = []
for cfg in configs:
    rs = runs_of(cfg)
    t_end = max(r["cpu_cum"][-1] for r in rs)
    grid = np.linspace(0.0, t_end, GRID_POINTS)
    curves = []
    for r in rs:
        idx = np.searchsorted(r["cpu_cum"], grid, side="right") - 1
        c = r["err"][np.clip(idx, 0, None)]
        c = np.where(grid <= r["cpu_cum"][-1], c, np.nan)
        curves.append(c)
    C = np.array(curves)
    grid_frames.append(pd.DataFrame(dict(
        method=cfg["method"], params=cfg["params"],
        cpu_time_s=grid,
        err_mean=np.nanmean(C, 0),
        err_min=np.nanmin(C, 0),
        err_max=np.nanmax(C, 0),
        n_runs=np.sum(~np.isnan(C), 0),
    )))
grid_df = pd.concat(grid_frames, ignore_index=True)
grid_df.to_csv(f"{FOLDER}/error_vs_cpu_time_{TAG}.csv", index=False)

# =============================================================================
# Output 3: per-configuration summary (iterations and CPU time)
# =============================================================================
summary_rows = []
for cfg in configs:
    rs = runs_of(cfg)
    total_cpu = pd.Series([r["cpu_cum"][-1] for r in rs])
    iters = pd.Series([r["iters"] for r in rs])
    step_ms = pd.Series([1e3 * np.mean(np.diff(r["cpu_cum"])) for r in rs])
    final_err = pd.Series([r["err"][-1] for r in rs])

    t_tol, it_tol = [], []
    for r in rs:
        hit = np.nonzero(r["err"] <= ERR_TOL)[0]
        t_tol.append(r["cpu_cum"][hit[0]] if hit.size else np.nan)
        it_tol.append(hit[0] if hit.size else np.nan)
    t_tol = pd.Series(t_tol)
    it_tol = pd.Series(it_tol)

    summary_rows.append(dict(
        method=cfg["method"], params=cfg["params"], runs=len(rs),
        iters_mean=iters.mean(),
        cpu_total_mean_s=total_cpu.mean(), cpu_total_std_s=total_cpu.std(),
        cpu_per_iter_mean_ms=step_ms.mean(), cpu_per_iter_std_ms=step_ms.std(),
        final_err_mean=final_err.mean(), final_err_std=final_err.std(),
        frac_reached_tol=t_tol.notna().mean(),
        cpu_to_tol_mean_s=t_tol.mean(),
        iters_to_tol_mean=it_tol.mean(),
    ))
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(f"{FOLDER}/summary_{TAG}.csv", index=False)
print("\n", summary_df.to_string(index=False))

# =============================================================================
# Plots
# =============================================================================
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'tab:orange', 'tab:purple', 'tab:brown']

fig1, ax1 = plt.subplots(figsize=(8, 5))
ax1.set_xlim(left=0)
for c, cfg in enumerate(configs):
    E = pad_stack([r["err"] for r in runs_of(cfg)])
    it = np.arange(E.shape[1])
    lbl = f'{cfg["method"]} {cfg["params"]}'
    col = colors[c % len(colors)]
    ax1.semilogy(it, np.nanmean(E, 0), color=col, label=lbl)
    ax1.fill_between(it, np.nanmin(E, 0), np.nanmax(E, 0), color=col, alpha=0.15)
ax1.set_xlabel("Iteration")
ax1.set_ylabel("Error ||x - x*|| (log scale)")
ax1.legend()
fig1.tight_layout()
fig1.savefig(f"{FOLDER}/error_vs_iter_{TAG}.png", dpi=150)

fig2, ax2 = plt.subplots(figsize=(8, 5))
for c, cfg in enumerate(configs):
    g = grid_df[(grid_df.method == cfg["method"]) & (grid_df.params == cfg["params"])]
    lbl = f'{cfg["method"]} {cfg["params"]}'
    col = colors[c % len(colors)]
    ax2.semilogy(g.cpu_time_s, g.err_mean, color=col, label=lbl)
    ax2.fill_between(g.cpu_time_s, g.err_min, g.err_max, color=col, alpha=0.15)
ax2.set_xlabel("CPU time (s)")
ax2.set_ylabel("Error ||x - x*|| (log scale)")
ax2.legend()
ax2.set_xlim(left=0)
if CPU_XMAX is not None:
    ax2.set_xlim(right=CPU_XMAX)
fig2.tight_layout()
fig2.savefig(f"{FOLDER}/error_vs_cpu_{TAG}.png", dpi=150)

if SHOW_PLOTS:
    plt.show()