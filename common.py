"""common.py -- parameters, crowd kernel, AS/agent dynamics, scenario, true world.

AS: second-order unicycle of Remark 3.8 in the equivalent polar state z = (q1, q2, theta, s),
s = ||v|| >= 0, smooth at rest.  Agents: v_k-dot = f_self(C_1(x, x_k)) + f_crowd(C_K(x_k, x)); in the true
world, beyond the band edges |q2| = band, a soft wall brings their normal velocity continuously to rest and
back to its mirror value (no velocity jump; not part of any method's model).
The crowd formulas take the math module M (numpy or casadi): simulator, closure and optimal
control problems evaluate literally the same expressions.
"""
import numpy as np


class P:
    T, dt_sim, N = 35.0, 0.02, 20                 # upper bound T_max of the free final time, world step, knots
    rho, u2_max, s_max = 5.0, 1.5, 4.0            # |u1| <= rho (braking (4)), |u2|, speed limit
    Delta_safe, eta, v_agent = 1.0, 3.5, 1.5      # safety (5) and agent speed bound (3)
    delta_min, delta_max, w = 0.05, 0.10, 0.05    # Algorithm 1 and control weight w
    lam = 0.01                                    # time weight: cost dist(x(T),X_goal)^2 + int (lam + w|u|^2)
    H_c = 4.0                                     # [s] constraint horizon of the baselines' clearance
    H_max = 8.0                                   # [s] window of the free final time: T <= t_i + H_max
    C_cm, tau0, R = 1.5, 3.0, 0.4                 # power law C_cm exp(-tau/tau0)/tau^2, contact radius
    tau_min, a_max, R_c = 0.25, 0.1, 2.0          # regularisation (Sec. 2): tau floor, cap, cut-off
    A_s, B_s, R_s = 1.2, 1.5, 5.0                 # f_self: the AS repels agents
    h_s, h_p = 0.05, 0.25                         # closure: |v_a| grid step, spacing of p in the ball
    y_start, y_goal, goal_rad = -8.0, 8.0, 0.8    # 16 m crossing, goal disc
    band, corr, ped_speed = 5.0, 15.0, 1.1        # |q2| <= band at t = 0, periodic |q1| <= corr
    A_w, w_wall = 1.5, 0.1                        # soft walls at |q2| = band: restoring acceleration, ramp depth
    budget = 10.0                                 # [s] maximal budget per resolution, all methods
    rt_budget = 0.1                               # [s] real-time budget = delta_max (reference line in fig_scaling)
    tol = 0.01                                    # [m] audit: smaller violations of (5) not counted
    Ks = [2, 5, 10, 25, 50, 75, 100, 125, 150]
    seeds = list(range(10))


def step(t, M=np):
    """C^1 smoothstep: 0 for t <= 0, 1 for t >= 1."""
    t = M.fmin(M.fmax(t, 0), 1)
    return t*t*(3 - 2*t)


def pair_force(x0, x1, v0, v1, M=np):
    """Force on agent a from agent j, x = q_j - q_a, v = v_a - v_j: exactly -grad_{q_a} of
    E(tau) = C_cm exp(-tau/tau0)/tau^2 (Karamouzas et al. 2014), tau the time to contact.  Sec. 2
    requires f_crowd bounded, Lipschitz and compactly supported: smooth gates (contact predicted,
    and in the future), a floor on tau, a cut-off at R_c and a smooth cap of the magnitude at a_max."""
    A, B = v0*v0 + v1*v1 + 1e-9, x0*v0 + x1*v1
    d = M.sqrt(x0*x0 + x1*x1 + 1e-9)
    disc = B*B - A*(d*d - P.R**2)
    sq = M.sqrt(M.fmax(disc, 0) + 1e-9)
    tr = (B - sq)/A
    gate = step(disc/(0.25*A*P.R**2), M)*step(tr/P.tau_min, M)*step((P.R_c - d)/0.4, M)
    tau = M.fmax(tr, P.tau_min)
    m = gate*P.C_cm*M.exp(-tau/P.tau0)/tau**2*(2/tau + 1/P.tau0)/sq
    fx, fy = -m*(x0 - tau*v0), -m*(x1 - tau*v1)
    c = P.a_max/(P.a_max + M.sqrt(fx*fx + fy*fy + 1e-12))
    return c*fx, c*fy


def f_self(y0, y1, M=np):
    """Force on an agent from the AS, y = q_AS - q_k: exponential repulsion cut off at R_s."""
    d = M.sqrt(y0*y0 + y1*y1 + 1e-9)
    m = -P.A_s*M.exp(-d/P.B_s)*step((P.R_s - d)/0.4, M)/d
    return m*y0, m*y1


def as_vel(z):
    return z[3]*np.array([np.cos(z[2]), np.sin(z[2])])


def arrived(z):
    return np.hypot(z[0], z[1] - P.y_goal) <= P.goal_rad


def scenario(K, rng):
    """AS at rest below the band; K pedestrians streaming along q1 inside the band |q2| <= band."""
    Q = np.c_[rng.uniform(-P.corr, P.corr, K), rng.uniform(-P.band, P.band, K)]
    V = np.c_[rng.choice([-1.0, 1.0], K)*P.ped_speed*rng.uniform(0.85, 1.15, K), rng.normal(0, 0.05, K)]
    return np.array([0.0, P.y_start, np.pi/2, 0.0]), Q, V


def agent_acc(Q, V, q):
    """f_self + f_crowd for every agent; minimum image along the periodic q1."""
    X, W = Q[None] - Q[:, None], V[:, None] - V[None]          # [a, j]: q_j - q_a and v_a - v_j
    X[..., 0] = (X[..., 0] + P.corr) % (2*P.corr) - P.corr
    fx, fy = pair_force(X[..., 0], X[..., 1], W[..., 0], W[..., 1])
    sx, sy = f_self(q[0] - Q[:, 0], q[1] - Q[:, 1])
    return np.c_[fx.sum(1) - np.diag(fx) + sx, fy.sum(1) - np.diag(fy) + sy]


def wall(Q):
    """Soft walls: beyond |q2| = band, a restoring acceleration ramping (C^1) up to A_w over w_wall.  It is
    conservative, so a pedestrian touching a wall stops, then regains its normal velocity mirrored, with no
    jump in velocity (it replaces the specular reflection of the previous version)."""
    d = np.abs(Q[:, 1]) - P.band
    return np.c_[np.zeros(len(Q)), -np.sign(Q[:, 1])*P.A_w*step(d/P.w_wall)]


def step_world(z, u, Q, V, h):
    """RK2 step of the true coupled system: periodic in q1, soft walls at |q2| = band; (3) is enforced by
    projecting |v_k| onto v_agent (the number of projected agents is returned); braking stops the AS, it
    never reverses."""
    def f(z, Q, V):
        return (np.array([z[3]*np.cos(z[2]), z[3]*np.sin(z[2]), u[1], u[0]]), V,
                agent_acc(Q, V, z[:2]) + wall(Q))
    dz, dQ, dV = f(z, Q, V)
    zh = z + h/2*dz
    zh[3] = max(zh[3], 0)
    dz, dQ, dV = f(zh, Q + h/2*dQ, V + h/2*dV)
    z, Q, V = z + h*dz, Q + h*dQ, V + h*dV
    z[3] = max(z[3], 0)
    Q[:, 0] = (Q[:, 0] + P.corr) % (2*P.corr) - P.corr
    n = np.linalg.norm(V, axis=1)
    return z, Q, V*np.minimum(1, P.v_agent/np.maximum(n, 1e-12))[:, None], int((n > P.v_agent).sum())


def nearest(z, Q, V):
    """k_i of (14): among the closest agents, one of the fastest nearing; and whether it is unique."""
    y = z[:2] - Q
    d2 = (y*y).sum(1)
    I = np.flatnonzero(d2 == d2.min())
    r = 2*(y[I]*(as_vel(z) - V[I])).sum(1)
    J = I[r == r.min()]
    return J[0], len(J) == 1


def persists(z, Q, V, k, dl):
    """k is predicted (constant velocities) to remain the nearest agent on [t_i, t_i + dl]."""
    return all(np.argmin(((z[:2] + t*as_vel(z) - Q - t*V)**2).sum(1)) == k for t in dl*np.arange(1, 6)/5)


def clearance(z0, Q0, z1, Q1, ds):
    """Worst instant of a world step for (5): distance and requirement at the SAME instant (5 instants,
    linear motion, exact speed max(s0 + c ds, 0) with ds = u1 h, so a stopped AS requires no clearance).
    Returns (fraction of the step, min_k ||q - q_k||, Delta_safe chi_eta(||v||))."""
    Q0 = Q0 + np.c_[2*P.corr*np.round((Q1[:, 0] - Q0[:, 0])/(2*P.corr)), 0*Q0[:, 1]]   # undo q1 wrap
    c = np.linspace(0, 1, 5)
    d = np.array([np.linalg.norm((1 - a)*(z0[:2] - Q0) + a*(z1[:2] - Q1), axis=1).min() for a in c])
    r = P.Delta_safe*np.minimum(np.maximum(z0[3] + c*ds, 0), P.eta)
    i = np.argmin(d - r)
    return c[i], d[i], r[i]
