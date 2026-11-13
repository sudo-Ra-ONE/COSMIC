import os
import sys
# Single-threaded BLAS / XLA, so that CPU time (summed over all threads of the process) is comparable between methods.
# These only take effect if they are set BEFORE numpy / jax are imported, and setdefault does not override values
# that are already set in the environment - both cases are detected and reported.
_PRELOADED = [mod for mod in ("numpy", "jax") if mod in sys.modules]


def _pin_to_one_core():
    """Restrict this process (and every thread it starts: XLA, BLAS, the QP solver) to ONE CPU core, so that CPU time and
    wall-clock time agree and CPU times are comparable between methods.  The XLA_FLAGS below are not sufficient for this
    in recent JAX versions.  Core: PICNN_CPU if set, otherwise the allowed core that is most idle right now.
    Set PICNN_PIN=0 to disable.  Linux only (elsewhere: start the script with an external tool such as `taskset`)."""
    if os.environ.get("PICNN_PIN", "1") != "1" or not hasattr(os, "sched_setaffinity"):
        return None
    allowed = sorted(os.sched_getaffinity(0))
    core = os.environ.get("PICNN_CPU")
    if core is None:
        def idle():
            out = {}
            with open("/proc/stat") as fh:
                for line in fh:
                    if line.startswith("cpu") and line[3].isdigit():
                        parts = line.split()
                        out[int(parts[0][3:])] = int(parts[4]) + int(parts[5])
            return out
        try:
            import time as _t
            a = idle(); _t.sleep(0.25); b = idle()
            core = max(allowed, key=lambda c: b.get(c, 0) - a.get(c, 0))
        except Exception:
            core = allowed[0]
    os.sched_setaffinity(0, {int(core)})
    return int(core)


PINNED_CORE = _pin_to_one_core()
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
TIMING_WARNINGS = []
if _PRELOADED:
    TIMING_WARNINGS.append(f"{' and '.join(_PRELOADED)} were imported before this script set the thread limits (notebook / IDE "
                           "session?), so they may run multi-threaded; start from a fresh terminal with `python file.py`")
if os.environ["OMP_NUM_THREADS"] != "1" or "intra_op_parallelism_threads=1" not in os.environ["XLA_FLAGS"]:
    TIMING_WARNINGS.append(f"thread limits are overridden by the environment (OMP_NUM_THREADS={os.environ['OMP_NUM_THREADS']}, "
                           f"XLA_FLAGS={os.environ['XLA_FLAGS']!r}); CPU times will be summed over several threads")
if PINNED_CORE is None:
    TIMING_WARNINGS.append("the process is not pinned to one CPU core (PICNN_PIN=0 or not Linux), so JAX may use several "
                           "threads and CPU times are then summed over them")
else:
    print(f"Process pinned to CPU core {PINNED_CORE} (single-threaded timing)")
for _w in TIMING_WARNINGS:
    print("TIMING WARNING:", _w)

import time
import pickle
import warnings
from functools import partial

import numpy as np
import pandas as pd
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.ndimage import zoom
from scipy.optimize import minimize
from sklearn.datasets import load_digits
from scipy.sparse.linalg import cg as sp_cg, LinearOperator
try:                                   # only needed for QP_SOLVER = "gurobi"
    import gurobipy as gp
    from gurobipy import GRB
except Exception:                      # pragma: no cover
    gp = GRB = None

warnings.filterwarnings("ignore", category=RuntimeWarning)


def _env(name, default):
    v = os.environ.get("PICNN_" + name)
    return default if v is None else type(default)(v)


# =============================================================================
# Settings
# =============================================================================
FOLDER = os.environ.get("PICNN_FOLDER", "/home/rawan/COSMIC/results/icnn_olivetti_completion_st")
TAG = os.environ.get("PICNN_TAG", "PICNN_Olivetti_Paper")
SEED = 0

# --- data --------------------------------------------------------------------
DATA_SOURCE = os.environ.get("PICNN_DATA", "olivetti")   # "olivetti" | "digits" (8x8 digits upsampled; smoke tests only)
OLIVETTI_NPY = None        # optional: path to an olivetti_faces.npy (400,64,64); None -> fetch_olivetti_faces()
N_TEST_SUBJECTS = 4        # subjects held out for the final comparison (disjoint from training and tuning)
HALF_H, HALF_W = 64, 32    # a half image: 64 rows x 32 columns (x = left half, y = right half)
n = HALF_H * HALF_W        # number of decision variables (2048)

# --- PICNN (architecture of the paper, Section 6.3) -------------------------------------------------------
CONV_CH = (32, 64, 64)     # 32x8x8 conv (stride 4x2), 64x4x4 conv (stride 2x2), 64x3x3 conv
FC_UNITS = 512             # 512 fully connected
ACT = "softplus"           # g in the output path: "softplus" (smooth, so the energy is twice differentiable) or "relu" (paper)
MU = 0.05                  # + 0.5*MU*||y||^2 makes F strongly convex in y -> unique minimiser
TRAIN_STEPS = _env("TRAIN_STEPS", 2000)
BATCH = 16
LR_TRAIN = 1e-3
DSM_SIGMA = (0.02, 0.25)   # noise levels of the denoising score matching loss
DSM_C = 0.25               # target gradient = DSM_C (y~ - y*), i.e. F is trained towards DSM_C/2 ||y - y*||^2 around the truth
FLIP_AUG = True            # random horizontal flip of the whole face during training
RETRAIN = False            # False: reuse the cached PICNN if it exists

# --- problems ----------------------------------------------------------------
N_PROBLEMS = _env("N_PROBLEMS", 3)     # test faces used for the final comparison
N_RUNS = _env("N_RUNS", 2)             # random initial points per problem (shared by all methods)
IC_LOW, IC_HIGH = -0.5, 1.5            # initial points are drawn over a WIDE range that also violates the box

# --- reference solution --------------------------------------------------------
REF_LBFGS_ITER = 5000      # L-BFGS-B iterations used to locate the reference solution and its active set
ACT_TOL = 1e-6             # a pixel is considered active if it is this close to its bound

# --- experiment --------------------------------------------------------------
N_ITER = _env("N_ITER", 3000)          # iterations per run (COSMIC, ADAM-AL, ADAM-PEN)
N_ITER_FL = _env("N_ITER_FL", 3000)     # FL solves a QP per iteration, so it gets fewer iterations
ERR_TOL = 1e-3
ERR_TOL_TIGHT = 1e-8
KKT_TOL = 1e-6
SAVE_EVERY = 10
GRID_POINTS = 400
N_TIME_ITER = _env("N_TIME_ITER", 500) # iterations of one stand-alone timing run (scan solvers); several seconds per run
TIME_REPEATS = _env("TIME_REPEATS", 5) # timing runs per configuration, interleaved over the configurations (best one is used)
SHOW_PLOTS = os.environ.get("PICNN_SHOW", "1") == "1"

# --- methods and hyperparameters (set by hand; there is no automatic tuning) -----------------
METHOD_ORDER = ["COSMIC", "FL", "ADAM-AL", "ADAM-PEN", "PGD-MOM"]   # remove a name to skip a method
# Hyperparameters of every method: edit freely.  An entry is either one dict or a LIST of dicts; with a list the method
# is run once per dict and every setting gets its own curve, CSV rows and summary line.
COSMIC_K_VALUES = [5.0]     # values of the COSMIC gain K to compare
HYPERPARAMS = {
    "COSMIC": dict(dt=0.3, K=5, beta=0.1, eps_sat=0.1),
    "FL": dict(eta=0.5, K=6.0),
    "ADAM-AL": dict(lr=1e-2, rho=10.0, inner=100, growth=1.03, decay=0.97),
    "ADAM-PEN": dict(lr=1e-2, rho=10.0, inner=100, growth=1.03, decay=0.97),
    "PGD-MOM": dict(lr=0.01, mom=0.9),
}

# COSMIC fixed settings
EPS_VAL = 1e-10
R_EPS, EPS_MIN, EPS_MAX, C_SHRINK = 1.05, 1e-16, 1e-1, 0.85

# Adam + quadratic penalty (ADAM-PEN): no multipliers,
#   P_rho(x) = f(x) + rho/2 * sum_i max(0, g_i)^2 ,  rho <- min(rho * growth, RHO_MAX) every `inner` steps
#   (penalty continuation); the multiplier estimate for the KKT metric is rho * max(0, g).
# Adam + augmented Lagrangian (PHR / method of multipliers)
#   L_rho(x, lam) = f(x) + 1/(2 rho) * sum_i ( max(0, lam_i + rho g_i)^2 - lam_i^2 )
#   inner: Adam steps on x for `inner` iterations, outer: lam <- max(0, lam + rho g(x)),
#   rho <- min(rho * growth, RHO_MAX), lr <- max(lr * decay, LR_MIN)
RHO_MAX, LR_MIN = 1e4, 1e-7
RESET_ADAM = True      # reset Adam moments after each multiplier update
B1, B2, ADAM_EPS = 0.9, 0.999, 1e-8

QP_SOLVER = os.environ.get("PICNN_QP", "gurobi")   # QP solver for FL: "gurobi" (as in First_Order_Opt.ipynb; the QP has
                        # m = 4096 variables, which needs a full licence - the restricted pip licence stops at 2000) or
                        # "clarabel" (pip install clarabel; interior point, tolerance 1e-12, no licence needed)
QP_RIDGE = 1e-10        # FL: tiny ridge so the rank-deficient J J^T is accepted as PSD
CAP = 1e12              # non-finite values are stored as this in the CSV files

os.makedirs(FOLDER, exist_ok=True)

# =============================================================================
# Data: x = left half, y = right half
# =============================================================================
rng_data = np.random.default_rng(SEED)
if DATA_SOURCE == "olivetti":
    if OLIVETTI_NPY is not None:
        faces = np.load(OLIVETTI_NPY).astype(float)
        subject = np.repeat(np.arange(len(faces) // 10), 10)           # 10 consecutive images per subject
    else:
        from sklearn.datasets import fetch_olivetti_faces
        d = fetch_olivetti_faces(shuffle=False)
        faces, subject = d.images.astype(float), d.target
    faces = np.clip(faces, 0.0, 1.0)
    subj = rng_data.permutation(np.unique(subject))
    is_te = np.isin(subject, subj[:N_TEST_SUBJECTS])
    F_train, F_test = faces[~is_te], faces[is_te]
    print(f"Olivetti faces: {len(F_train)} train / {len(F_test)} test images "
          f"({40 - N_TEST_SUBJECTS}/{N_TEST_SUBJECTS} subjects), 64x64 pixels")
else:                                                                    # stand-in for smoke tests only
    dg = load_digits().images / 16.0
    dg = np.clip(np.array([zoom(im, 8.0, order=3) for im in dg]), 0.0, 1.0)
    perm = rng_data.permutation(len(dg))
    F_test, F_train = dg[perm[:60]], dg[perm[60:]]
    print(f"DIGITS STAND-IN (not faces): {len(F_train)} train / {len(F_test)} test, 64x64")


def halves(F):
    """(N, 64, 64) -> left halves (N, 64, 32) and flattened right halves (N, 2048)."""
    return F[:, :, :HALF_W].copy(), F[:, :, HALF_W:].reshape(len(F), -1).copy()


X_train, Y_train = halves(F_train)
X_test, Y_test = halves(F_test)

# =============================================================================
# Convolutional PICNN with the architecture of the paper (convex in y)
# =============================================================================
ZKEYS = ("Wz1", "Wz2", "Wz3", "Wz4")      # the weights W^z that must stay >= 0


def conv(x, W, stride=(1, 1)):
    return jax.lax.conv_general_dilated(x[None], W, stride, "SAME", dimension_numbers=("NHWC", "HWIO", "NHWC"))[0]


def init_picnn(key):
    c1, c2, c3 = CONV_CH
    flat = 8 * 8 * c3                      # 4096 after the third convolution (SAME padding, total stride 8 x 4)
    ks = iter(jax.random.split(key, 40))
    f32 = jnp.float32

    def dense(fan_in, fan_out, scale=1.0):
        return scale * jax.random.normal(next(ks), (fan_in, fan_out), f32) / float(np.sqrt(fan_in))

    def kern(kh, kw, cin, cout, scale=1.0):
        return scale * jax.random.normal(next(ks), (kh, kw, cin, cout), f32) / float(np.sqrt(kh * kw * cin))

    def nonneg(shape, fan_in):             # small non-negative initial W^z (keeps z of the order of one)
        return 0.8 * jnp.abs(jax.random.normal(next(ks), shape, f32)) / fan_in

    z = lambda *s: jnp.zeros(s, f32)
    o = lambda *s: jnp.ones(s, f32)
    return dict(
        # input path  u_1 .. u_4  (x -> 32x8x8 -> 64x4x4 -> 64x3x3 -> 512)
        Wt1=kern(8, 8, 1, c1), bt1=z(c1), Wt2=kern(4, 4, c1, c2), bt2=z(c2), Wt3=kern(3, 3, c2, c3), bt3=z(c3),
        Wt4=dense(flat, FC_UNITS), bt4=z(FC_UNITS),
        # y gate and layer 1 of the output path
        Wyu0=0.1 * kern(3, 3, 1, 1), byu0=o(1), Wy0=kern(8, 8, 1, c1), Wu0=kern(8, 8, 1, c1), b0=z(c1),
        # layer 2:  z_1 (16x16x32) -> z_2 (8x8x64)
        Wzu1=0.1 * kern(1, 1, c1, c1), bzu1=o(c1), Wz1=nonneg((4, 4, c1, c2), 16 * c1), Wy1=0.3 * kern(8, 8, 1, c2),
        Wu1=kern(4, 4, c1, c2), b1=z(c2),
        # layer 3:  z_2 -> z_3 (8x8x64)
        Wzu2=0.1 * kern(1, 1, c2, c2), bzu2=o(c2), Wz2=nonneg((3, 3, c2, c3), 9 * c2), Wy2=0.3 * kern(8, 8, 1, c3),
        Wu2=kern(3, 3, c2, c3), b2=z(c3),
        # layer 4:  z_3 -> z_4 (512, fully connected)
        Wzu3=0.1 * kern(1, 1, c3, c3), bzu3=o(c3), Wz3=nonneg((flat, FC_UNITS), flat), Wy3=0.3 * dense(n, FC_UNITS),
        Wu3=dense(flat, FC_UNITS), b3=z(FC_UNITS),
        # output layer (scalar)
        Wzu4=0.1 * dense(FC_UNITS, FC_UNITS), bzu4=o(FC_UNITS), Wz4=nonneg((FC_UNITS, 1), FC_UNITS),
        Wy4=0.05 * dense(n, 1), Wu4=0.05 * dense(FC_UNITS, 1), b4=z(1))


def g_act(v):
    return jax.nn.softplus(v) if ACT == "softplus" else jax.nn.relu(v)


def picnn(pr, y):
    """pr = dict(net=<parameters>, u=<left half x, shape (64, 32, 1)>);  y: flattened right half (2048,).
    Returns F(x, y) = f_theta(x, y) + MU/2 ||y||^2, convex in y."""
    p, x = pr["net"], pr["u"]
    Y = y.reshape(HALF_H, HALF_W, 1)
    # input path
    u1 = jax.nn.relu(conv(x, p["Wt1"], (4, 2)) + p["bt1"])                  # 16 x 16 x 32
    u2 = jax.nn.relu(conv(u1, p["Wt2"], (2, 2)) + p["bt2"])                 # 8 x 8 x 64
    u3 = jax.nn.relu(conv(u2, p["Wt3"]) + p["bt3"])                         # 8 x 8 x 64
    u4 = jax.nn.relu(u3.reshape(-1) @ p["Wt4"] + p["bt4"])                  # 512
    # output path
    gy = conv(x, p["Wyu0"]) + p["byu0"]                                     # gate of y (any sign: y enters linearly)
    z1 = g_act(conv(Y * gy, p["Wy0"], (4, 2)) + conv(x, p["Wu0"], (4, 2)) + p["b0"])
    g1 = jax.nn.relu(conv(u1, p["Wzu1"]) + p["bzu1"])
    z2 = g_act(conv(z1 * g1, p["Wz1"], (2, 2)) + conv(Y, p["Wy1"], (8, 4)) + conv(u1, p["Wu1"], (2, 2)) + p["b1"])
    g2 = jax.nn.relu(conv(u2, p["Wzu2"]) + p["bzu2"])
    z3 = g_act(conv(z2 * g2, p["Wz2"]) + conv(Y, p["Wy2"], (8, 4)) + conv(u2, p["Wu2"]) + p["b2"])
    g3 = jax.nn.relu(conv(u3, p["Wzu3"]) + p["bzu3"])
    z4 = g_act((z3 * g3).reshape(-1) @ p["Wz3"] + y @ p["Wy3"] + u3.reshape(-1) @ p["Wu3"] + p["b3"])
    g4 = jax.nn.relu(u4 @ p["Wzu4"] + p["bzu4"])
    out = (z4 * g4) @ p["Wz4"] + y @ p["Wy4"] + u4 @ p["Wu4"] + p["b4"]
    return out[0] + 0.5 * MU * jnp.dot(y, y)


icnn = picnn                                      # the solvers below call icnn / grad_icnn
grad_icnn = jax.grad(icnn, argnums=1)


def hvp_icnn(pr, x, v):                           # Hessian-vector product (forward over reverse)
    return jax.jvp(lambda z: grad_icnn(pr, z), (x,), (v,))[1]


tmap = jax.tree_util.tree_map


# --- training: denoising score matching, float32, Adam, clipping of W^z >= 0 after each step --------------
def dsm_loss(params, Xc, Yt, Yn):
    gr = jax.vmap(lambda x, y: grad_icnn(dict(net=params, u=x), y))(Xc, Yn)
    return jnp.mean(jnp.sum((gr - DSM_C * (Yn - Yt)) ** 2, axis=1)) / n * 100.0


@jax.jit
def train_step(params, m1, v1, t, Xc, Yt, Yn, lr):
    loss, g = jax.value_and_grad(dsm_loss)(params, Xc, Yt, Yn)
    m1 = tmap(lambda a, b_: B1 * a + (1 - B1) * b_, m1, g)
    v1 = tmap(lambda a, b_: B2 * a + (1 - B2) * b_ ** 2, v1, g)
    new = tmap(lambda p, a, b_: p - lr * (a / (1 - B1 ** t)) / (jnp.sqrt(b_ / (1 - B2 ** t)) + ADAM_EPS), params, m1, v1)
    new = {k: (jnp.maximum(v, 0.0) if k in ZKEYS else v) for k, v in new.items()}        # projection onto W^z >= 0
    return new, m1, v1, loss


def train_batch(rng):
    idx = rng.integers(0, len(F_train), BATCH)
    Fb = F_train[idx]
    if FLIP_AUG:
        flip = rng.random(BATCH) < 0.5
        Fb = np.where(flip[:, None, None], Fb[:, :, ::-1], Fb)
    Xb, Yb = halves(Fb)
    sig = rng.uniform(*DSM_SIGMA, (BATCH, 1))
    Yn = Yb + sig * rng.normal(size=Yb.shape)
    f32 = np.float32
    return Xb[..., None].astype(f32), Yb.astype(f32), Yn.astype(f32)


cache = f"{FOLDER}/picnn_params_{TAG}_{DATA_SOURCE}_s{TRAIN_STEPS}.pkl"
if os.path.exists(cache) and not RETRAIN:
    with open(cache, "rb") as fh:
        PARAMS32 = tmap(jnp.asarray, pickle.load(fh))
    print(f"Loaded cached PICNN from {cache}")
else:
    PARAMS32 = init_picnn(jax.random.PRNGKey(SEED))
    print(f"PICNN with {sum(int(np.prod(a.shape)) for a in PARAMS32.values()):,} parameters "
          f"({sum(int(np.prod(PARAMS32[k].shape)) for k in ZKEYS):,} constrained to be >= 0)")
    mom, vel = tmap(jnp.zeros_like, PARAMS32), tmap(jnp.zeros_like, PARAMS32)
    rng_tr = np.random.default_rng(SEED + 1)
    t_train, run_loss = time.time(), []
    for t in range(1, TRAIN_STEPS + 1):
        Xb, Yb, Ynb = train_batch(rng_tr)
        lr_t = LR_TRAIN * (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * (t - 1) / max(TRAIN_STEPS, 1))))    # cosine decay to 10 %
        PARAMS32, mom, vel, loss = train_step(PARAMS32, mom, vel, np.float32(t), Xb, Yb, Ynb, np.float32(lr_t))
        run_loss.append(float(loss))
        if t == 1 or t % max(1, TRAIN_STEPS // 10) == 0:
            print(f"  PICNN train step {t:5d}  dsm loss={np.mean(run_loss[-50:]):.5f}   ({time.time() - t_train:.0f} s)")
    print(f"PICNN training took {time.time() - t_train:.1f} s")
    with open(cache, "wb") as fh:
        pickle.dump(tmap(np.asarray, PARAMS32), fh)

PARAMS = tmap(lambda a: jnp.asarray(a, jnp.float64), PARAMS32)       # the solvers work in float64
assert all(float(jnp.min(PARAMS[k])) >= 0.0 for k in ZKEYS), "W^z must be non-negative"

f_val = jax.jit(icnn)
f_grad = jax.jit(grad_icnn)
f_valgrad = jax.jit(jax.value_and_grad(icnn, argnums=1))
f_hvp = jax.jit(hvp_icnn)

# convexity / strong-convexity check:  v' H v >= MU ||v||^2 for random directions at random points
_rc = np.random.default_rng(11)
_chk = []
for _ in range(4):
    _pr = dict(net=PARAMS, u=jnp.asarray(X_test[_rc.integers(len(X_test))][..., None]))
    _y, _v = jnp.asarray(_rc.uniform(-0.5, 1.5, n)), jnp.asarray(_rc.normal(size=n))
    _chk.append(float(_v @ f_hvp(_pr, _y, _v)) / float(_v @ _v))
print(f"convexity check  v'Hv/|v|^2 at 4 random (y, v): min = {min(_chk):.4f}  (>= MU = {MU} expected)")


# =============================================================================
# Completion problems:  g(y) = A y - b <= 0 with A = [I; -I], b = [1; 0]  (pixel-wise bounds 0 <= y <= 1)
# =============================================================================
# The constraint matrix A = [P; -P; I; -I] is never formed inside the JAX solvers: they receive P (here an
# empty 0 x n matrix, the observation constraints being merged into the pixel bounds) and use these
# structured products (cost O(n)).
def m_of(P):
    return 2 * P.shape[0] + 2 * P.shape[1]


def Aop(P, x):                     # A x
    y = P @ x
    return jnp.concatenate([y, -y, x, -x])


def ATop(P, v):                    # A' v
    p, nn = P.shape
    return P.T @ (v[:p] - v[p:2 * p]) + v[2 * p:2 * p + nn] - v[2 * p + nn:]


def make_problem(y_true, x_left):
    """One completion problem: the left half x_left (64, 32) is observed, the right half y (2048,) is the unknown."""
    hi, lo = np.ones(n), np.zeros(n)
    b = np.concatenate([hi, -lo])
    u = x_left[..., None]
    P = np.zeros((0, n))
    return dict(x_true=y_true, x_left=x_left, miss=np.ones(n, bool), P=P, b=b, m=2 * n, u=u, lo=lo, hi=hi,
                A_sp=sp.vstack([sp.identity(n), -sp.identity(n)], format="csr"),
                pr=dict(net=PARAMS, u=jnp.asarray(u)), P_j=jnp.asarray(P), b_j=jnp.asarray(b), u_j=jnp.asarray(u))


def metrics_full(x, lam, A, b, params, x_ref, f_ref):
    """(error to reference, KKT residual, max constraint violation, objective gap).  A here is P."""
    g = Aop(A, x) - b
    gf = grad_icnn(params, x)
    stat = jnp.linalg.norm(gf + ATop(A, lam))
    feas = jnp.maximum(jnp.max(g), 0.0)
    comp = jnp.max(jnp.abs(lam * g))
    dual = jnp.maximum(jnp.max(-lam), 0.0)
    kkt = jnp.max(jnp.stack([stat, feas, comp, dual]))
    return jnp.linalg.norm(x - x_ref), kkt, feas, icnn(params, x) - f_ref


metrics_one = jax.jit(metrics_full)


def kkt_of(x, lam, prob):
    return float(metrics_one(jnp.asarray(x), jnp.asarray(lam), prob["P_j"], prob["b_j"], prob["pr"],
                             jnp.zeros(n), 0.0)[1])


def solve_reference(prob):
    """Strongly convex problem -> unique minimiser, computed independently of the compared methods:
      1. L-BFGS-B (scipy, bound constraints) from the mid-gray image locates the solution and its active set;
      2. an active-set Newton method polishes it: pixels at a bound are fixed, the free pixels solve grad_F = 0 by
         Newton's method (matrix-free CG with exact Hessian-vector products), and the active set is corrected
         (a free pixel leaving [lo, hi] is fixed at the bound, a fixed pixel whose multiplier would be negative is released)
         until it no longer changes.
    For pixel bounds the multipliers are explicit (lam_up = max(0,-grad), lam_lo = max(0,grad) on the active
    pixels), so the result is certified by its KKT residual."""
    pr, lo, hi = prob["pr"], prob["lo"], prob["hi"]

    def fun(y):
        v, g = f_valgrad(pr, jnp.asarray(y))
        return float(v), np.asarray(g, dtype=float)

    res = minimize(fun, np.full(n, 0.5), jac=True, method="L-BFGS-B", bounds=list(zip(lo, hi)),
                   options=dict(maxiter=REF_LBFGS_ITER, maxfun=4 * REF_LBFGS_ITER, maxcor=30, ftol=1e-15, gtol=1e-11))
    xs = np.clip(res.x, lo, hi)

    def multipliers(x, up, dn):
        gf = np.asarray(f_grad(pr, x))
        return np.concatenate([np.where(up, np.maximum(-gf, 0.0), 0.0), np.where(dn, np.maximum(gf, 0.0), 0.0)])

    lam_s = multipliers(xs, (hi - xs) < ACT_TOL, (xs - lo) < ACT_TOL)
    kkt_s = kkt_of(xs, lam_s, prob)

    pin_up, pin_dn = lam_s[:n] > 1e-9, lam_s[n:] > 1e-9
    x = xs.copy()
    for _ in range(20):                                       # active-set rounds
        free = ~(pin_up | pin_dn)
        x[pin_up], x[pin_dn] = hi[pin_up], lo[pin_dn]
        for _ in range(10):                                   # Newton on the free pixels, grad_F(x) = 0
            gf = np.asarray(f_grad(pr, x))

            def mv(v):
                w = np.zeros(n)
                w[free] = v
                return np.asarray(f_hvp(pr, x, jnp.asarray(w)))[free]

            d, _ = sp_cg(LinearOperator((int(free.sum()),) * 2, matvec=mv), -gf[free], rtol=1e-14, atol=0.0, maxiter=3000)
            x[free] += d
            if np.linalg.norm(d) < 1e-14:
                break
        gf = np.asarray(f_grad(pr, x))
        new_up, new_dn = free & (x > hi + 1e-12), free & (x < lo - 1e-12)          # left the box -> fix at the bound
        rel_up, rel_dn = pin_up & (gf > 1e-12), pin_dn & (gf < -1e-12)             # multiplier would be negative -> release
        if not (new_up.any() or new_dn.any() or rel_up.any() or rel_dn.any()):
            break
        pin_up, pin_dn = (pin_up | new_up) & ~rel_up, (pin_dn | new_dn) & ~rel_dn
    free = ~(pin_up | pin_dn)
    lam_p = multipliers(x, pin_up, pin_dn)
    ok = (np.all(x >= lo - 1e-11) and np.all(x <= hi + 1e-11) and np.all(np.isfinite(x)))
    kkt_p = kkt_of(x, lam_p, prob) if ok else np.inf
    if kkt_p < kkt_s:
        x_ref, kkt_ref, how = x, kkt_p, "L-BFGS-B + active-set Newton-CG"
    else:
        x_ref, kkt_ref, how = xs, kkt_s, "L-BFGS-B"
    return x_ref, float(f_val(pr, x_ref)), kkt_ref, True, int((~free).sum()), how


def build_problems(images, rng, label, check_unique=False):
    probs = []
    for pi, ti in enumerate(images):
        prob = make_problem(label_set[label][1][ti], label_set[label][0][ti])
        prob["x_ref"], prob["f_ref"], kkt_ref, ok, n_act, how = solve_reference(prob)
        mse_ref = np.mean((prob["x_ref"] - prob["x_true"])[prob["miss"]] ** 2)
        mirror = prob["x_left"][:, ::-1].reshape(-1)                      # naive baselines: mirrored left half, mean right half
        psnr = lambda v: -10 * np.log10(np.mean((v - prob["x_true"]) ** 2))
        print(f"{label} problem {pi}: m = {prob['m']}, f* = {prob['f_ref']:.10f}, pixels at a bound = {n_act}, "
              f"reference KKT residual = {kkt_ref:.1e} ({how}); PSNR of the completed half = {psnr(prob['x_ref']):.2f} dB "
              f"(mirrored left half {psnr(mirror):.2f} dB, mean training right half {psnr(Y_train.mean(0)):.2f} dB)")
        if kkt_ref > 1e-9:
            print("  warning: reference KKT residual is above 1e-9; errors below that level are not meaningful")
        probs.append(prob)
    return probs


def pack(probs, x0s):
    """Stack problems x runs into batch arrays (problem-major)."""
    idx = [(pi, r) for pi in range(len(probs)) for r in range(len(x0s[pi]))]
    return dict(
        idx=idx,
        x0=jnp.asarray(np.array([x0s[pi][r] for pi, r in idx])),
        A=jnp.asarray(np.array([probs[pi]["P"] for pi, r in idx])),      # NB: the batch stores P, not A
        U=jnp.asarray(np.array([probs[pi]["u"] for pi, r in idx])),      # context of the PICNN
        b=jnp.asarray(np.array([probs[pi]["b"] for pi, r in idx])),
        xref=jnp.asarray(np.array([probs[pi]["x_ref"] for pi, r in idx])),
        fref=jnp.asarray(np.array([probs[pi]["f_ref"] for pi, r in idx])))


def pick_runs(batch, sel):
    return {k: (v if k == "idx" else v[sel]) for k, v in batch.items()}


# =============================================================================
# Solvers.  COSMIC and the Adam / projected-gradient methods are expressed as (init, step, view) triples and run inside
# jax.lax.scan (vmapped over problems x initial points), so that
#   * the hyperparameters are traced arguments: changing them does not change the compiled code,
#   * the CPU time of the algorithm itself is measured without Python dispatch overhead.
# =============================================================================
def phi_fn(g, xi, eps):
    return xi - g - jnp.sqrt(g ** 2 + xi ** 2 + eps ** 2)


def sat(v, eps_sat):
    return jnp.clip(v / eps_sat, -1.0, 1.0)


def eps_update(eps_vec, phi_val):
    shrink_mask = jnp.abs(phi_val) < C_SHRINK * eps_vec
    return jnp.where(shrink_mask, jnp.maximum(eps_vec / R_EPS, EPS_MIN), eps_vec)


# ---- COSMIC (matrix-free: J = A is constant, M and G are diagonal) -----------
def cosmic_init(x0, A, b, hp):
    return x0, jnp.zeros(m_of(A)), jnp.full(m_of(A), EPS_VAL)


def cosmic_step(state, A, b, params, hp):
    x, xi, eps_vec = state
    dt, K, beta, eps_sat = hp["dt"], hp["K"], hp["beta"], hp["eps_sat"]
    delta_M = dt * K / 2.0
    g = Aop(A, x) - b
    x_dot = -grad_icnn(params, x) - ATop(A, xi)
    x_next = x + dt * x_dot
    phi_val = phi_fn(g, xi, eps_vec)
    s = jnp.sqrt(g ** 2 + xi ** 2 + eps_vec ** 2)
    d = 1.0 - xi / s
    d_tilde = jnp.sqrt(d ** 2 + delta_M ** 2)
    xi_dot = (-K * phi_val - beta * sat(phi_val, eps_sat) + (1.0 + g / s) * Aop(A, x_dot)) / d_tilde
    xi_next = xi + dt * xi_dot
    return x_next, xi_next, eps_update(eps_vec, phi_val)


def cosmic_view(state, A, b, hp):
    return state[0], state[1]


# ---- Adam + augmented Lagrangian --------------------------------------------
def al_init(x0, A, b, hp):
    z = jnp.zeros_like(x0)
    return x0, z, z, jnp.zeros(()), jnp.zeros(m_of(A)), hp["rho"], hp["lr"], jnp.zeros((), jnp.int32)


def al_step(state, A, b, params, hp):
    x, m1, v1, t, lam, rho, lr, k = state
    t = t + 1.0
    g = Aop(A, x) - b
    mult = jnp.maximum(0.0, lam + rho * g)                # gradient of the PHR penalty term
    grad = grad_icnn(params, x) + ATop(A, mult)
    m1 = B1 * m1 + (1 - B1) * grad
    v1 = B2 * v1 + (1 - B2) * grad ** 2
    x = x - lr * (m1 / (1 - B1 ** t)) / (jnp.sqrt(v1 / (1 - B2 ** t)) + ADAM_EPS)
    k = k + 1
    outer = (k % hp["inner"]) == 0                        # multiplier update every `inner` Adam steps
    lam = jnp.where(outer, jnp.maximum(0.0, lam + rho * (Aop(A, x) - b)), lam)
    rho = jnp.where(outer, jnp.minimum(rho * hp["growth"], RHO_MAX), rho)
    lr = jnp.where(outer, jnp.maximum(lr * hp["decay"], LR_MIN), lr)
    if RESET_ADAM:
        m1 = jnp.where(outer, 0.0, m1)
        v1 = jnp.where(outer, 0.0, v1)
        t = jnp.where(outer, 0.0, t)
    return x, m1, v1, t, lam, rho, lr, k


def al_view(state, A, b, hp):
    x, lam, rho = state[0], state[4], state[5]
    return x, jnp.maximum(0.0, lam + rho * (Aop(A, x) - b))   # effective multiplier, used for the KKT metric


# ---- Adam + quadratic penalty with continuation ----------------------------------
def pen_step(state, A, b, params, hp):
    x, m1, v1, t, lam, rho, lr, k = state                 # lam stays 0 (kept so the state matches the AL one)
    t = t + 1.0
    mult = rho * jnp.maximum(0.0, Aop(A, x) - b)
    grad = grad_icnn(params, x) + ATop(A, mult)
    m1 = B1 * m1 + (1 - B1) * grad
    v1 = B2 * v1 + (1 - B2) * grad ** 2
    x = x - lr * (m1 / (1 - B1 ** t)) / (jnp.sqrt(v1 / (1 - B2 ** t)) + ADAM_EPS)
    k = k + 1
    outer = (k % hp["inner"]) == 0
    rho = jnp.where(outer, jnp.minimum(rho * hp["growth"], RHO_MAX), rho)
    lr = jnp.where(outer, jnp.maximum(lr * hp["decay"], LR_MIN), lr)
    if RESET_ADAM:
        m1 = jnp.where(outer, 0.0, m1)
        v1 = jnp.where(outer, 0.0, v1)
        t = jnp.where(outer, 0.0, t)
    return x, m1, v1, t, lam, rho, lr, k


def pen_view(state, A, b, hp):
    x, rho = state[0], state[5]
    return x, rho * jnp.maximum(0.0, Aop(A, x) - b)           # penalty multiplier estimate

# ---- Projected gradient descent with momentum (inference method of the ICNN paper; uses the box structure) -------
#   v <- mom v - lr grad F(y),   y <- clip(y + v, lo, hi).   Iterates are feasible after the first step.
#   Multipliers for the KKT metric: lam_up = max(0, -grad F) on pixels at the upper bound, lam_lo = max(0, grad F) at the lower.
def pgd_init(x0, A, b, hp):
    return x0, jnp.zeros_like(x0)


def pgd_step(state, A, b, params, hp):
    x, v = state
    nn = x.shape[0]
    hi, lo = b[-2 * nn:-nn], -b[-nn:]
    v = hp["mom"] * v - hp["lr"] * grad_icnn(params, x)
    return jnp.clip(x + v, lo, hi), v


def pgd_view(state, A, b, hp, params):
    x = state[0]
    nn = x.shape[0]
    hi, lo = b[-2 * nn:-nn], -b[-nn:]
    gf = grad_icnn(params, x)
    lam_box = jnp.concatenate([jnp.where(x >= hi - 1e-12, jnp.maximum(-gf, 0.0), 0.0),
                               jnp.where(x <= lo + 1e-12, jnp.maximum(gf, 0.0), 0.0)])
    return x, jnp.concatenate([jnp.zeros(m_of(A) - 2 * nn), lam_box])


SOLVERS = {"COSMIC": (cosmic_init, cosmic_step, cosmic_view),
           "ADAM-AL": (al_init, al_step, al_view),
           "ADAM-PEN": (al_init, pen_step, pen_view),
           "PGD-MOM": (pgd_init, pgd_step, pgd_view)}


@partial(jax.jit, static_argnames=("method", "N", "log"))
def run_batch(method, x0, A, b, xref, fref, U, params, hp, N, log):
    """Run `method` for N iterations on a batch of (problem, initial point) pairs.
    Returns the final iterate and, if log, (err, kkt, viol, gap) arrays of shape (B, N+1)."""
    init, step, view = SOLVERS[method]

    def one(x0i, Ai, bi, xri, fri, Ui):
        pr = dict(net=params, u=Ui)
        s0 = init(x0i, Ai, bi, hp)

        def rec(s):
            xx, ll = view(s, Ai, bi, hp, pr) if method == "PGD-MOM" else view(s, Ai, bi, hp)
            return metrics_full(xx, ll, Ai, bi, pr, xri, fri)

        def body(s, _):
            s1 = step(s, Ai, bi, pr, hp)
            return s1, (rec(s1) if log else None)

        sN, ys = jax.lax.scan(body, s0, None, length=N)
        if log:
            m0 = rec(s0)
            ys = tuple(jnp.concatenate([a[None], y]) for a, y in zip(m0, ys))
        return sN[0], ys

    return jax.vmap(one)(x0, A, b, xref, fref, U)


def to_hp(method, cfg):
    hp = {k: jnp.asarray(v, dtype=jnp.int32 if k == "inner" else jnp.float64) for k, v in cfg.items()}
    return hp


# ---- Feedback linearisation (FL), from SQP_ineq: lambda from a QP (Gurobi), then
#      x <- x - eta (grad f + J^T lambda)
def solve_lambda(J, grad_f, h_x, K_gain, Q=None):
    """min 0.5 l'(J J')l + (J grad_f - K h)'l  s.t. l >= 0  (non-Hessian branch of solve_lambda).
    J is constant for linear constraints, so the caller may pass the precomputed Q = J J'.
    Threads=1 keeps the CPU-time measurement from being spread over Gurobi worker threads."""
    m_ = J.shape[0]
    if Q is None:
        Q = J @ J.T
    Q = (0.5 * (Q + Q.T) + QP_RIDGE * sp.identity(m_, format="csc")).tocsc()   # sparse: J = A is very sparse
    c = J @ grad_f - K_gain * h_x
    if QP_SOLVER == "clarabel":
        import clarabel
        Qs = sp.triu(Q).tocsc()
        st = clarabel.DefaultSettings()
        st.verbose = False
        st.tol_gap_abs = st.tol_gap_rel = st.tol_feas = 1e-12
        sol = clarabel.DefaultSolver(Qs, c, -sp.identity(m_, format="csc"), np.zeros(m_),
                                     [clarabel.NonnegativeConeT(m_)], st).solve()
        if str(sol.status) != "Solved" and not getattr(solve_lambda, "warned", False):
            print("flag: Clarabel status", sol.status, "(printed once; usually a diverging iterate)")
            solve_lambda.warned = True
        return np.maximum(np.array(sol.x), 0.0)
    model = gp.Model()
    model.setParam("OutputFlag", 0)
    model.setParam("Threads", 1)
    lam = model.addMVar(shape=m_, name="lambda", lb=0.0)
    model.setMObjective(0.5 * Q, c, 0.0, lam, lam, lam, GRB.MINIMIZE)
    model.optimize()
    if model.status != GRB.OPTIMAL and not getattr(solve_lambda, "warned", False):
        print("flag: Gurobi status", model.status, "(printed once; usually a diverging iterate)")
        solve_lambda.warned = True
    out = np.array(lam.X) if model.SolCount > 0 else np.zeros(m_)
    model.dispose()
    return out


def run_fl(prob, x0, eta, K_gain, N):
    A, b = prob["A_sp"], prob["b"]
    x = np.asarray(x0, dtype=float).copy()
    log = dict(err=[], kkt=[], viol=[], gap=[], cpu_step=[], wall_step=[])
    Q_fl = (A @ A.T).tocsc()                         # constant for linear constraints
    for k in range(N + 1):
        t0, w0 = time.process_time(), time.perf_counter()
        grad_f = np.asarray(f_grad(prob["pr"], x))
        h_x = A @ x - b
        lam = solve_lambda(A, grad_f, h_x, K_gain, Q_fl)
        x_next = x - eta * (grad_f + A.T @ lam)
        dt_k, dw_k = time.process_time() - t0, time.perf_counter() - w0
        e, kk, vi, ga = metrics_one(jnp.asarray(x), jnp.asarray(lam), prob["P_j"], prob["b_j"], prob["pr"],
                                    jnp.asarray(prob["x_ref"]), prob["f_ref"])
        log["err"].append(float(e)); log["kkt"].append(float(kk))
        log["viol"].append(float(vi)); log["gap"].append(float(ga))
        if k == N:
            break
        log["cpu_step"].append(dt_k)
        log["wall_step"].append(dw_k)
        x = x_next
        if not np.all(np.isfinite(x)) or np.abs(x).max() > 1e6:      # diverged: stop, pad with CAP
            pad = N - k
            for key in ("err", "kkt", "viol", "gap"):
                log[key] += [CAP] * pad
            log["cpu_step"] += [dt_k] * (len(log["err"]) - 1 - len(log["cpu_step"]))
            break
    return log, x


# =============================================================================
# Helpers
# =============================================================================
def clean(a):
    return np.nan_to_num(np.asarray(a, dtype=float), nan=CAP, posinf=CAP, neginf=CAP).clip(-CAP, CAP)


def label_of(method, cfg):
    return ",".join(f"{k}={v:.3g}" if isinstance(v, (float, np.floating)) else f"{k}={v}"
                    for k, v in cfg.items())


# =============================================================================
# Problems and initial points
# =============================================================================
rng_prob = np.random.default_rng(SEED + 2)
label_set = dict(eval=(X_test, Y_test))
problems = build_problems(rng_prob.choice(len(Y_test), N_PROBLEMS, replace=False), rng_prob, "eval")

rng_init = np.random.default_rng(SEED + 3)
x0_lists = [[rng_init.uniform(IC_LOW, IC_HIGH, n) for _ in range(N_RUNS)] for _ in problems]


eval_batch = pack(problems, x0_lists)
n_eval = len(eval_batch["idx"])

# =============================================================================
# Experiments (hand-set hyperparameters)
# =============================================================================
configs = [dict(method=m, tag="manual", params=label_of(m, hp), cfg=dict(hp))
           for m in METHOD_ORDER for hp in (HYPERPARAMS[m] if isinstance(HYPERPARAMS[m], (list, tuple)) else [HYPERPARAMS[m]])]
pd.DataFrame([dict(method=c["method"], params=c["params"], **{f"hp_{k}": v for k, v in c["cfg"].items()})
              for c in configs]).to_csv(f"{FOLDER}/hyperparameters_{TAG}.csv", index=False)
for c in configs:
    c["name"] = f"{c['method']} {c['params']}"


def finalize(err, kkt, viol, gap, cpu_step, prob, x_final):
    err, kkt, viol, gap = clean(err), clean(kkt), clean(viol), clean(gap)
    cpu_cum = np.concatenate([[0.0], np.cumsum(cpu_step)])
    return dict(err=err, kkt=kkt, viol=viol, gap=gap, cpu_cum=cpu_cum, iters=len(err) - 1,
                mse_truth=float(np.mean((np.nan_to_num(x_final, nan=CAP) - prob["x_true"])[prob["miss"]] ** 2)), x_final=x_final)


def time_scan_solvers(cfg_list):
    """CPU and wall-clock seconds per iteration of the compiled solvers alone (no metric logging), on one problem.
    All configurations are timed back to back and the TIME_REPEATS repetitions are interleaved over the configurations,
    so that every method sees the same machine state; the fastest repetition is used.  Returns
    {name: dict(cpu, wall, spread)} with spread = (slowest - fastest) / fastest over the repetitions."""
    one = pick_runs(eval_batch, slice(0, 1))
    args = (one["x0"], one["A"], one["b"], one["xref"], one["fref"], one["U"])
    hps = {c["name"]: to_hp(c["method"], c["cfg"]) for c in cfg_list}
    for c in cfg_list:                                                        # compile + warm-up
        jax.block_until_ready(run_batch(c["method"], *args, PARAMS, hps[c["name"]], N_TIME_ITER, False)[0])
    cpu = {c["name"]: [] for c in cfg_list}
    wall = {c["name"]: [] for c in cfg_list}
    for _ in range(TIME_REPEATS):
        for c in cfg_list:
            t0, w0 = time.process_time(), time.perf_counter()
            jax.block_until_ready(run_batch(c["method"], *args, PARAMS, hps[c["name"]], N_TIME_ITER, False)[0])
            cpu[c["name"]].append((time.process_time() - t0) / N_TIME_ITER)
            wall[c["name"]].append((time.perf_counter() - w0) / N_TIME_ITER)
    out = {}
    for name in cpu:
        i = int(np.argmin(cpu[name]))
        out[name] = dict(cpu=cpu[name][i], wall=wall[name][i], spread=(max(cpu[name]) - min(cpu[name])) / min(cpu[name]))
    return out


print(f"\nTiming the compiled solvers ({TIME_REPEATS} x {N_TIME_ITER} iterations each, interleaved)")
TIMING = time_scan_solvers([c for c in configs if c["method"] != "FL"])
for name, t in TIMING.items():
    print(f"  {name[:60]:60s} cpu/iter = {1e3 * t['cpu']:7.3f} ms   wall/iter = {1e3 * t['wall']:7.3f} ms   "
          f"cpu/wall = {t['cpu'] / t['wall']:4.2f}   spread over repetitions = {100 * t['spread']:4.0f} %")
if TIMING:
    if max(t["cpu"] / t["wall"] for t in TIMING.values()) > 1.25:
        TIMING_WARNINGS.append("CPU time exceeds wall-clock time by more than 25 %: several threads are running and their "
                               "CPU times are summed (check OMP_NUM_THREADS / XLA_FLAGS, or pin with `taskset -c 0`)")
    if max(t["spread"] for t in TIMING.values()) > 0.30:
        TIMING_WARNINGS.append("repeated timings of the same solver differ by more than 30 %: the machine is loaded or changing "
                               "clock frequency; rerun on an idle machine before using the CPU times")

results = []
print("\nFinal comparison")
for cfg in configs:
    method = cfg["method"]
    if method == "FL":
        for pi, prob in enumerate(problems):
            for run, x0 in enumerate(x0_lists[pi]):
                log, xf = run_fl(prob, x0, cfg["cfg"]["eta"], cfg["cfg"]["K"], N_ITER_FL)
                res = finalize(log["err"], log["kkt"], log["viol"], log["gap"], log["cpu_step"], prob, xf)
                results.append(dict(name=cfg["name"], method=method, tag=cfg["tag"], params=cfg["params"],
                                    problem=pi, run=run, wall_it=float(np.mean(log["wall_step"])), spread=np.nan, **res))
    else:
        n_it = N_ITER
        xf_all, (err, kkt, viol, gap) = run_batch(method, eval_batch["x0"], eval_batch["A"], eval_batch["b"],
                                                  eval_batch["xref"], eval_batch["fref"], eval_batch["U"], PARAMS,
                                                  to_hp(method, cfg["cfg"]), n_it, True)
        err, kkt, viol, gap, xf_all = map(np.asarray, (err, kkt, viol, gap, xf_all))
        tm = TIMING[cfg["name"]]                                                # same cost for every run
        cpu_it = tm["cpu"]
        for bi, (pi, run) in enumerate(eval_batch["idx"]):
            res = finalize(err[bi], kkt[bi], viol[bi], gap[bi], np.full(n_it, cpu_it), problems[pi], xf_all[bi])
            results.append(dict(name=cfg["name"], method=method, tag=cfg["tag"], params=cfg["params"],
                                problem=pi, run=run, wall_it=tm["wall"], spread=tm["spread"], **res))
    rs = [r for r in results if r["name"] == cfg["name"]]
    print(f"{cfg['name'][:70]:70s} cpu/iter={1e3 * np.mean([np.diff(r['cpu_cum']).mean() for r in rs]):7.4f} ms  "
          f"final err={np.mean([r['err'][-1] for r in rs]):.2e}  kkt={np.mean([r['kkt'][-1] for r in rs]):.2e}  "
          f"viol={np.mean([r['viol'][-1] for r in rs]):.1e}  "
          f"PSNR(truth)={np.mean([-10 * np.log10(r['mse_truth']) for r in rs]):.2f} dB")


def runs_of(cfg):
    return [r for r in results if r["name"] == cfg["name"]]


def pad_stack(arrs):
    L = max(len(a) for a in arrs)
    out = np.full((len(arrs), L), np.nan)
    for i, a in enumerate(arrs):
        out[i, :len(a)] = a
    return out


def on_cpu_grid(rs, key, grid):
    curves = []
    for r in rs:
        idx = np.searchsorted(r["cpu_cum"], grid, side="right") - 1
        c = r[key][np.clip(idx, 0, None)]
        curves.append(np.where(grid <= r["cpu_cum"][-1], c, np.nan))
    return np.array(curves)


# =============================================================================
# Output 1: per-iteration table of every run (iteration index AND CPU time)
# =============================================================================
rows = []
for r in results:
    L = len(r["err"])
    idx = np.unique(np.r_[np.arange(0, L, SAVE_EVERY), L - 1])
    step_s = np.r_[np.diff(r["cpu_cum"]), np.nan]
    rows.append(pd.DataFrame(dict(
        method=r["method"], tag=r["tag"], params=r["params"], problem=r["problem"], run=r["run"],
        iter=idx, cpu_cum_s=r["cpu_cum"][idx], cpu_step_s=step_s[idx],
        kkt=r["kkt"][idx], max_violation=r["viol"][idx], obj_gap=r["gap"][idx], err=r["err"][idx])))
pd.concat(rows, ignore_index=True).to_csv(f"{FOLDER}/iterations_{TAG}.csv", index=False)

# =============================================================================
# Output 2: mean / min / max over runs versus ITERATION (for the shaded mean plots)
#           and versus CPU time (common grid per configuration)
# =============================================================================
frames_it, frames_cpu, grids = [], [], {}
for cfg in configs:
    rs = runs_of(cfg)
    E, Kk, V = (pad_stack([r[k] for r in rs]) for k in ("err", "kkt", "viol"))
    it = np.arange(E.shape[1])
    sel = np.unique(np.r_[np.arange(0, E.shape[1], SAVE_EVERY), E.shape[1] - 1])
    frames_it.append(pd.DataFrame(dict(
        method=cfg["method"], tag=cfg["tag"], params=cfg["params"], iter=it[sel],
        err_mean=np.nanmean(E, 0)[sel], err_min=np.nanmin(E, 0)[sel], err_max=np.nanmax(E, 0)[sel],
        kkt_mean=np.nanmean(Kk, 0)[sel], kkt_min=np.nanmin(Kk, 0)[sel], kkt_max=np.nanmax(Kk, 0)[sel],
        viol_mean=np.nanmean(V, 0)[sel], viol_max=np.nanmax(V, 0)[sel])))
    grid = np.linspace(0.0, max(r["cpu_cum"][-1] for r in rs), GRID_POINTS)
    Ec, Kc = on_cpu_grid(rs, "err", grid), on_cpu_grid(rs, "kkt", grid)
    grids[cfg["name"]] = dict(grid=grid, E=Ec, K=Kc)
    frames_cpu.append(pd.DataFrame(dict(
        method=cfg["method"], tag=cfg["tag"], params=cfg["params"], cpu_time_s=grid,
        err_mean=np.nanmean(Ec, 0), err_min=np.nanmin(Ec, 0), err_max=np.nanmax(Ec, 0),
        kkt_mean=np.nanmean(Kc, 0), kkt_min=np.nanmin(Kc, 0), kkt_max=np.nanmax(Kc, 0),
        n_runs=np.sum(~np.isnan(Ec), 0))))
pd.concat(frames_it, ignore_index=True).to_csv(f"{FOLDER}/error_vs_iter_{TAG}.csv", index=False)
pd.concat(frames_cpu, ignore_index=True).to_csv(f"{FOLDER}/error_vs_cpu_time_{TAG}.csv", index=False)

# =============================================================================
# Output 3: per-configuration summary
# =============================================================================
def first_hit(arr, tol):
    hit = np.nonzero(arr <= tol)[0]
    return hit[0] if hit.size else None


summary_rows = []
for cfg in configs:
    rs = runs_of(cfg)
    step_ms = pd.Series([1e3 * np.mean(np.diff(r["cpu_cum"])) for r in rs])
    row = dict(method=cfg["method"], tag=cfg["tag"], params=cfg["params"], runs=len(rs),
               iters=rs[0]["iters"], cpu_per_iter_mean_ms=step_ms.mean(), cpu_per_iter_std_ms=step_ms.std(),
               wall_per_iter_mean_ms=1e3 * np.mean([r["wall_it"] for r in rs]),
               cpu_wall_ratio=step_ms.mean() / (1e3 * np.mean([r["wall_it"] for r in rs])),
               timing_spread_pct=100 * np.mean([r["spread"] for r in rs]),
               cpu_total_mean_s=np.mean([r["cpu_cum"][-1] for r in rs]),
               n_diverged=int(sum(r["err"].max() > 1e3 for r in rs)),
               final_err_mean=np.mean([r["err"][-1] for r in rs]), final_err_std=np.std([r["err"][-1] for r in rs]),
               final_kkt_mean=np.mean([r["kkt"][-1] for r in rs]),
               final_viol_mean=np.mean([r["viol"][-1] for r in rs]),
               final_obj_gap_mean=np.mean([r["gap"][-1] for r in rs]),
               psnr_truth_mean_db=np.mean([-10 * np.log10(r["mse_truth"]) for r in rs]))
    for label, key, tol in [("err_tol", "err", ERR_TOL), ("err_tight", "err", ERR_TOL_TIGHT), ("kkt_tol", "kkt", KKT_TOL)]:
        hits = [first_hit(r[key], tol) for r in rs]
        ok = [h is not None for h in hits]
        row[f"frac_reached_{label}"] = float(np.mean(ok))
        row[f"iters_to_{label}_mean"] = float(np.mean([h for h in hits if h is not None])) if any(ok) else np.nan
        row[f"cpu_to_{label}_mean_s"] = float(np.mean([r["cpu_cum"][h] for r, h in zip(rs, hits) if h is not None])) if any(ok) else np.nan
    summary_rows.append(row)
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(f"{FOLDER}/summary_{TAG}.csv", index=False)
print("\n", summary_df[["method", "params", "iters", "cpu_per_iter_mean_ms", "wall_per_iter_mean_ms", "cpu_wall_ratio",
                        "timing_spread_pct", "n_diverged", "final_err_mean",
                        "final_kkt_mean", "final_viol_mean", "psnr_truth_mean_db", "iters_to_kkt_tol_mean",
                        "cpu_to_kkt_tol_mean_s"]].to_string(index=False, float_format=lambda v: f"{v:.3g}"))
if float(summary_df["cpu_wall_ratio"].max()) > 1.25 and not any("exceeds wall-clock" in w for w in TIMING_WARNINGS):
    TIMING_WARNINGS.append("CPU time exceeds wall-clock time by more than 25 % for at least one method (FL: Gurobi / the QP solver "
                           "may be using several threads)")
for _w in TIMING_WARNINGS:
    print("TIMING WARNING:", _w)
if not TIMING_WARNINGS:
    print("Timing check passed: single-threaded (CPU ~ wall-clock) and repeatable.")

# =============================================================================
# Plots
# =============================================================================
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'tab:orange', 'tab:purple', 'tab:brown', 'tab:pink',
          'tab:gray', 'tab:olive', 'tab:cyan']


def plot_set(cfg_list):
    fig1, ax1 = plt.subplots(1, 2, figsize=(13, 5))
    for c, cfg in enumerate(cfg_list):
        rs = runs_of(cfg)
        for ax, key in zip(ax1, ("err", "kkt")):
            M = pad_stack([r[key] for r in rs])
            it = np.arange(M.shape[1])
            ax.semilogy(it, np.clip(np.nanmean(M, 0), 1e-17, CAP), color=colors[c % len(colors)], label=cfg["name"][:48])
            ax.fill_between(it, np.clip(np.nanmin(M, 0), 1e-17, CAP), np.clip(np.nanmax(M, 0), 1e-17, CAP),
                            color=colors[c % len(colors)], alpha=0.15)
    ax1[0].set_ylabel("Error ||x - x*|| (log scale)")
    ax1[1].set_ylabel("KKT residual (log scale)")
    for ax in ax1:
        ax.set_xlabel("Iteration")
        ax.set_xscale("symlog", linthresh=10)
        ax.set_ylim(1e-16, 1e3)
        ax.grid(alpha=0.3)
    ax1[0].legend(fontsize=6)
    fig1.tight_layout()
    fig1.savefig(f"{FOLDER}/convergence_vs_iter_{TAG}.png", dpi=150)
    for key, ylabel, fname in [("E", "Error ||x - x*|| (log scale)", "error_vs_cpu"),
                               ("K", "KKT residual (log scale)", "kkt_vs_cpu")]:
        fig, ax = plt.subplots(figsize=(8, 5))
        for c, cfg in enumerate(cfg_list):
            G = grids[cfg["name"]]
            ax.semilogy(G["grid"], np.clip(np.nanmean(G[key], 0), 1e-17, CAP), color=colors[c % len(colors)],
                        label=cfg["name"][:48])
            ax.fill_between(G["grid"], np.clip(np.nanmin(G[key], 0), 1e-17, CAP),
                            np.clip(np.nanmax(G[key], 0), 1e-17, CAP), color=colors[c % len(colors)], alpha=0.15)
        ax.set_xlabel("CPU time (s)")
        ax.set_ylabel(ylabel)
        ax.set_ylim(1e-16, 1e3)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6)
        fig.tight_layout()
        fig.savefig(f"{FOLDER}/{fname}_{TAG}.png", dpi=150)


plot_set(configs)

# Completed faces: ground truth, observed left half, reference minimiser, every method (first initial point)
best = {(c["method"] if sum(q["method"] == c["method"] for q in configs) == 1 else c["name"]): c for c in configs}
METHODS = list(best)                              # one row per configuration (label = method, or method + parameters)
row_titles = ["ground truth", "observed (gray = unknown)", "reference argmin F"] + METHODS
fig3, axs = plt.subplots(len(row_titles), N_PROBLEMS, figsize=(2.2 * N_PROBLEMS + 2, 1.9 * len(row_titles)), squeeze=False)
full = lambda prob, y: np.concatenate([prob["x_left"], np.clip(np.nan_to_num(y), 0, 1).reshape(HALF_H, HALF_W)], axis=1)
for pi, prob in enumerate(problems):
    imgs = [full(prob, prob["x_true"]), full(prob, np.full(n, 0.5)), full(prob, prob["x_ref"])]
    for method in METHODS:
        r = [q for q in runs_of(best[method]) if q["problem"] == pi and q["run"] == 0][0]
        imgs.append(full(prob, r["x_final"]))
    for ri, img in enumerate(imgs):
        axs[ri, pi].imshow(img, cmap="gray", vmin=0, vmax=1)
        axs[ri, pi].set_xticks([])
        axs[ri, pi].set_yticks([])
        if pi == 0:
            axs[ri, pi].set_ylabel(row_titles[ri], fontsize=7)
fig3.tight_layout()
fig3.savefig(f"{FOLDER}/completed_faces_{TAG}.png", dpi=150)

if SHOW_PLOTS:
    plt.show()