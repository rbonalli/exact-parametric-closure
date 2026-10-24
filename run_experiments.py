#!/usr/bin/env python3
"""run_experiments.py -- Algorithm 1 vs Full OCP vs Truncated OCP (Section 4).

Identical for the three methods: IPOPT with the same options (exact Hessian) and a 10 s budget per
resolution; free final time T in [t_i + N delta_max, min(t_i + H_max, T_max)], N RK2 steps of (T - t_i)/N; cost
dist(x(T), X_goal)^2 + int_{t_i}^T (lam + w ||u||^2); hard constraints only (Solve_Succeeded);
||v|| <= 4 m/s; warm start from the previous solution; same observation and crowd for each seed.
Run: python build_closure.py && python run_experiments.py
Outputs: fig_safety.png, fig_scaling.png, fig_timeline_<method>_K<K>.png, tables.txt, results.pkl
Redraw figures and tables from results.pkl without re-running: python run_experiments.py plots
"""


import pickle
import sys
import time
import numpy as np
import casadi as ca
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import P, pair_force, f_self, scenario, step_world, nearest, persists, clearance, arrived, as_vel

CL = np.load("closure.npz")
SG, TAB, MG = CL["s_grid"], CL["T"], CL["M"].max()     # |v_a| grid, f_hat_G(s, p_j), sup of one force
QG = np.array([0.0, P.y_goal])
OPTS = {"ipopt": {"print_level": 0, "sb": "yes", "max_cpu_time": P.budget}, "print_time": False,
        "expand": True}                                  # one flat SX graph per problem: faster evaluations
NAMES = {"ours": "Algorithm 1 (this work)", "full": "Full OCP", "trunc": r"Truncated OCP ($\lceil \ln K \rceil$ nearest)"}
COL = {"ours": "C0", "full": "C3", "trunc": "C2"}


def M_dmax(K):
    """M_delta_max of (13), with C_agent = ||f_self|| + ||f_crowd|| = A_s + (K-1) M_G."""
    return P.rho*P.Delta_safe - P.v_agent - (P.A_s + (K - 1)*MG)*P.delta_max


def rk2(f, x, u, h):
    return x + h*f(x + h/2*f(x, u), u)


def f_as(z, u):
    return ca.vertcat(z[3]*ca.cos(z[2]), z[3]*ca.sin(z[2]), u[1], u[0])


def f_base(z, u, m):
    """AS and m agents with their own dynamics f_self + f_crowd (the OCP of Section 3)."""
    qx, qy, vx, vy = (z[4 + i*m:4 + (i + 1)*m] for i in range(4))
    D = lambda a: ca.repmat(a.T, m, 1) - ca.repmat(a, 1, m)                # [a, j] = a_j - a_a
    fx, fy = pair_force(D(qx), D(qy), -D(vx), -D(vy), ca)
    sx, sy = f_self(z[0] - qx, z[1] - qy, ca)
    I = 1 - ca.DM.eye(m)
    return ca.vertcat(f_as(z, u), vx, vy, ca.sum2(fx*I) + sx, ca.sum2(fy*I) + sy)


def f_joint(x, u, Tj, Km1):
    """(AS, y, v_y) of OCP^i: AS dynamics and eq. (6) with the closure f_hat = (K-1) R(v_k)
    f_hat_G(|v_k|, p_j), linear interpolation on the uniform |v| grid (Tj: 2 x len(SG))."""
    e = ca.vertcat(ca.cos(x[2]), ca.sin(x[2]))
    vk = x[3]*e - x[6:8]                                                      # closest agent velocity
    sk = ca.sqrt(ca.sumsqr(vk) + 1e-9)
    a = ca.mtimes(Tj, ca.fmax(0, 1 - ca.fabs(ca.fmin(sk, P.v_agent) - ca.DM(SG))/(SG[1] - SG[0])))
    fh = Km1*ca.vertcat(vk[0]*a[0] - vk[1]*a[1], vk[1]*a[0] + vk[0]*a[1])/sk
    return ca.vertcat(x[3]*e, u[1], u[0], x[6:8],
                      u[0]*e + u[1]*x[3]*ca.vertcat(-e[1], e[0]) - ca.vertcat(*f_self(x[4], x[5], ca)) - fh)


def build(nz, npar, f, cons):
    """Shared transcription with free final time: decision variables X, U and the remaining duration
    T_f = T - t_i in [N delta_max, min(H_max, T_max - t_i)]: the first knot interval (constant control)
    always contains the committed [t_i, t_i + delta_i] and steps stay <= H_max/N.  cons(X, U, p, F) adds
    each method's constraints."""
    zs, us = ca.SX.sym("z", nz), ca.SX.sym("u", 2)
    F = ca.Function("f", [zs, us], [f(zs, us)])
    X, U, Tf, p = ca.MX.sym("X", nz, P.N + 1), ca.MX.sym("U", 2, P.N), ca.MX.sym("Tf"), ca.MX.sym("p", npar)
    g = [X[:, 0] - p[:nz]] + [X[:, k + 1] - rk2(F, X[:, k], U[:, k], Tf/P.N) for k in range(P.N)]
    gc, lc, uc = cons(X, U, p, F)
    J = (ca.fmax(ca.sqrt(ca.sumsqr(X[:2, P.N] - QG) + 1e-9) - P.goal_rad, 0)**2
         + P.lam*Tf + P.w*Tf/P.N*ca.sumsqr(U))
    S = ca.nlpsol("S", "ipopt", {"x": ca.veccat(X, U, Tf), "f": J, "g": ca.veccat(*g, *gc), "p": p}, OPTS)
    lo, hi = np.full(nz, -np.inf), np.full(nz, np.inf)
    lo[3], hi[3] = 0.0, P.s_max                                               # 0 <= ||v|| <= 4 m/s
    n0 = nz*(P.N + 1)
    return dict(S=S, nz=nz, lbx=np.r_[np.tile(lo, P.N + 1), np.tile([-P.rho, -P.u2_max], P.N), P.N*P.delta_max],
                ubx=np.r_[np.tile(hi, P.N + 1), np.tile([P.rho, P.u2_max], P.N), P.T],
                lbg=np.r_[np.zeros(n0), lc], ubg=np.r_[np.zeros(n0), uc])


def build_ours():
    """OCP^i_delta_i, p = (x(t_i), y(t_i), v_y(t_i), delta_i, K-1).  S for every p_j of the closure
    grid at t_i + delta_i/2 and t_i + delta_i (two RK2 half-steps under the first control: no knot
    lies inside [t_i, t_i + delta_i] since T_f/N >= delta_max; S at t_i only involves observed data and
    is checked before solving), RF at t_i + delta_i, its upper bound M_delta_max(K) set per K."""
    xs, us, ts, ks = ca.SX.sym("x", 8), ca.SX.sym("u", 2), ca.SX.sym("T", 2, len(SG)), ca.SX.sym("k")
    FJ = ca.Function("fj", [xs, us, ts, ks], [f_joint(xs, us, ts, ks)])

    def cons(X, U, p, F):
        g = []
        for j in range(TAB.shape[1]):
            xj, Tj = p[:8], ca.DM(np.ascontiguousarray(TAB[:, j, :].T))
            for _ in range(2):
                xj = rk2(lambda a, b: FJ(a, b, Tj, p[9]), xj, U[:, 0], p[8]/2)
                g.append(ca.sumsqr(xj[4:6]) - (P.Delta_safe*ca.fmin(xj[3], P.eta))**2)
        return g + [p[3] + p[8]*U[0, 0]], np.zeros(len(g) + 1), np.full(len(g) + 1, np.inf)
    return build(4, 10, f_as, cons)


def build_base(m):
    """Section 3's OCP via MPC with m agents (Full: m = K, Truncated: m = ceil(ln K)), state
    (q, theta, s, q_x, q_y, v_x, v_y) of size 4 + 4m.  Clearance to each of them, as S of Algorithm 1, at
    t_i + delta_max/2 and t_i + delta_max (two RK2 half-steps under the first control), and at every knot;
    run_base keeps only the knots within the constraint horizon H_c."""
    def clear(z):
        return (z[0] - z[4:4 + m])**2 + (z[1] - z[4 + m:4 + 2*m])**2 - (P.Delta_safe*ca.fmin(z[3], P.eta))**2

    def cons(X, U, p, F):
        z = [p]
        for _ in range(2):
            z.append(rk2(F, z[-1], U[:, 0], P.delta_max/2))
        g = [clear(x) for x in z[1:] + [X[:, k] for k in range(1, P.N + 1)]]
        return g, np.zeros(m*(P.N + 2)), np.full(m*(P.N + 2), np.inf)
    return build(4 + 4*m, 4 + 4*m, lambda z, u: f_base(z, u, m), cons)


def Tf_max(t):
    return max(P.N*P.delta_max, min(P.H_max, P.T - t))   # T <= min(t + H_max, T_max), except in the last 2 s


def solve(O, w0, p, t, ubg=None, lbg=None):
    ubx = O["ubx"].copy()
    ubx[-1] = Tf_max(t)
    t0 = time.perf_counter()
    r = O["S"](x0=w0, lbx=O["lbx"], ubx=ubx, lbg=O["lbg"] if lbg is None else lbg,
               ubg=O["ubg"] if ubg is None else ubg, p=p)
    st = O["S"].stats()["return_status"]
    return np.array(r["x"]).ravel(), st == "Solve_Succeeded", st, 1e3*(time.perf_counter() - t0)


def guess(O, z, w, t, Qs=None, Vs=None):
    """Warm start: previous solution (its duration already shortened by the elapsed time) with the
    observed AS state written in; first iteration: constant acceleration along the heading, reaching
    the edge of X_goal at T_f = H_max.  Agents: constant velocity from their observed states."""
    if w is None:
        tk, e = P.H_max/P.N*np.arange(P.N + 1)[:, None], np.array([np.cos(z[2]), np.sin(z[2])])
        D = max(np.hypot(z[0], z[1] - P.y_goal) - P.goal_rad, 0.0)
        a = np.clip(2*(D - z[3]*P.H_max)/P.H_max**2, -P.rho, P.rho)
        w = np.r_[np.c_[z[:2] + (z[3]*tk + a*tk**2/2)*e, np.full((P.N + 1, 1), z[2]), np.clip(z[3] + a*tk, 0, P.s_max),
                        np.zeros((P.N + 1, O["nz"] - 4))].ravel(), np.tile([a, 0.0], P.N), P.H_max]
    w = w.copy()
    w[-1] = np.clip(w[-1], P.N*P.delta_max, Tf_max(t))
    tk = w[-1]/P.N*np.arange(P.N + 1)[:, None]
    Z = w[:O["nz"]*(P.N + 1)].reshape(P.N + 1, O["nz"])
    Z[0, :4] = z
    if Qs is not None:
        Z[:, 4:] = np.c_[Qs[:, 0] + tk*Vs[:, 0], Qs[:, 1] + tk*Vs[:, 1],
                         np.tile(Vs[:, 0], (P.N + 1, 1)), np.tile(Vs[:, 1], (P.N + 1, 1))]
    return w


def new_log():
    return dict(solve=[], fallback=[], fine=[], traj=[], proj=0, halvings=0, Ju=0.0)


def advance(z, u, Q, V, span, t, L):
    """Implement u on [t, t + span] in the true world, auditing (5) at every world step."""
    n = max(1, round(span/P.dt_sim))
    for _ in range(n):
        z1, Q1, V1, k = step_world(z, u, Q, V, span/n)
        c, d, r = clearance(z, Q, z1, Q1, u[0]*span/n)
        L["fine"].append((t + c*span/n, d, r))
        L["proj"] += k
        z, Q, V, t = z1, Q1, V1, t + span/n
    L["Ju"] += P.w*span*float(u @ u)
    L["traj"].append((t, *z, Q.copy()))
    return z, Q, V, t


def finish(L, z, t, status):
    f = np.array(L["fine"]).reshape(-1, 3)
    L.update(status=status, t_end=t, reached=bool(arrived(z)), viol=int((f[:, 1] < f[:, 2] - P.tol).sum()),
             margin=(f[:, 1] - f[:, 2]).min() if len(f) else np.nan,
             J=max(np.hypot(z[0], z[1] - P.y_goal) - P.goal_rad, 0)**2 + P.lam*t + L["Ju"])
    return L


def run_ours(K, seed, O):
    """Algorithm 1, delta_i obtained by dichotomy as described in Section 4."""
    z, Q, V = scenario(K, np.random.default_rng(seed))
    ubg = O["ubg"].copy()
    ubg[-1] = M_dmax(K)                                                       # RF
    L, t, w = new_log(), 0.0, None
    while t < P.T and not arrived(z):
        k, unique = nearest(z, Q, V)                                          # line 3
        dl = P.delta_max if unique else 0.0                                   # lines 4-8
        while dl >= P.delta_min and not persists(z, Q, V, k, dl):
            dl /= 2
        y, vy = z[:2] - Q[k], as_vel(z) - V[k]
        if np.linalg.norm(y) < P.Delta_safe*min(z[3], P.eta):               # S at t_i
            dl = 0.0
        u, ms = None, np.nan
        while dl >= P.delta_min:                                              # lines 9-10
            x, ok, st, tm = solve(O, guess(O, z, w, t), np.r_[z, y, vy, dl, K - 1], t, ubg)
            ms = tm if np.isnan(ms) else ms + tm
            if tm > 1e3*P.budget:
                L["solve"].append(ms)
                return finish(L, z, t, "budget")
            if ok:
                w, u = x, x[4*(P.N + 1):4*(P.N + 1) + 2]
                break
            dl, L["halvings"] = dl/2, L["halvings"] + 1
        L["solve"].append(ms)
        L["fallback"].append(u is None)
        if u is None:                                                         # lines 11-12: brake at rate b_i
            m = np.hypot(*(z[:2] - Q).T).min() - P.Delta_safe*min(z[3], P.eta)     # clearance margin m_i
            b = min(P.rho, max(0.0, (P.v_agent + z[3] - m/P.delta_min)/P.Delta_safe))
            u, dl = np.array([-b, 0.0]), P.delta_min
        span = min(dl, P.T - t)                                               # line 14
        z, Q, V, t = advance(z, u, Q, V, span, t, L)
        if w is not None:
            w[-1] -= span
    return finish(L, z, t, "ok")


def run_base(K, seed, m, O):
    """Full (m = K) or Truncated (m = ceil(ln K)) OCP on [t_i, T], T free, clearance within H_c, re-planned
    every delta_max;
    the mission is abandoned at the first resolution that fails or exceeds the budget."""
    z, Q, V = scenario(K, np.random.default_rng(seed))
    L, t, w = new_log(), 0.0, None
    while t < P.T and not arrived(z):
        s = np.sort(np.argsort(((z[:2] - Q)**2).sum(1))[:m])                 # m nearest, index order
        w0, lbg = guess(O, z, w, t, Q[s], V[s]), O["lbg"].copy()             # constraint horizon H_c:
        lbg[O["nz"]*(P.N + 1) + m*(2 + int(P.H_c*P.N/w0[-1] + 1e-9)):] = -np.inf  # drop knots beyond t_i + H_c
        x, ok, st, tm = solve(O, w0, np.r_[z, Q[s].T.ravel(), V[s].T.ravel()], t, lbg=lbg)
        L["solve"].append(tm)
        if tm > 1e3*P.budget or not ok:
            return finish(L, z, t, "budget" if tm > 1e3*P.budget else st)
        w, span = x, min(P.delta_max, P.T - t)
        z, Q, V, t = advance(z, x[O["nz"]*(P.N + 1):][:2], Q, V, span, t, L)
        w[-1] -= span
    return finish(L, z, t, "ok")


def report(seed, K, n, L):
    s = np.array(L["solve"], float)
    q = np.percentile(s[np.isfinite(s)], [25, 50, 75, 100]) if np.isfinite(s).any() else [np.nan]*4
    print(f"seed {seed} K={K:3d} {n:5s} | {L['status'][:24]:24s} | reached {L['reached']!s:5s} t {L['t_end']:5.1f} s "
          f"J {L['J']:6.2f} | solve ms median {q[1]:7.1f} IQR [{q[0]:.0f}, {q[2]:.0f}] max {q[3]:7.1f} | "
          f"violations > {P.tol} m: {L['viol']}, worst margin {L['margin']:+.3f} m | projections {L['proj']} | "
          f"fallback {100*np.mean(L['fallback'] or [0]):4.1f} % halvings {L['halvings']}")


def timeline(L, n, K):
    """Snapshots of one mission: band where pedestrians stream at t = 0, AS path to come (grey) and
    done (blue), required clearance (red), goal (green); below, crossing progress and speed."""
    T_, Z = np.array([r[0] for r in L["traj"]]), np.array([r[1:5] for r in L["traj"]])
    idx, th = np.linspace(0, len(T_) - 1, 4).astype(int), np.linspace(0, 2*np.pi, 120)
    fig = plt.figure(figsize=(9, 6.6))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, .62], hspace=.42, wspace=.18)
    for i, j in enumerate(idx):
        a, Q, r = fig.add_subplot(gs[i // 2, i % 2]), L["traj"][j][5], P.Delta_safe*min(Z[j, 3], P.eta)
        a.axhspan(-P.band, P.band, color="C1", alpha=.07)
        a.plot(Q[:, 0], Q[:, 1], "o", ms=3, color="0.55", zorder=2)
        a.plot(Z[:, 0], Z[:, 1], color="0.8", lw=2, zorder=1)
        a.plot(Z[:j + 1, 0], Z[:j + 1, 1], color="C0", lw=2, zorder=3)
        a.plot(Z[j, 0], Z[j, 1], "o", color="C0", ms=5, zorder=5)
        a.plot(Z[j, 0] + r*np.cos(th), Z[j, 1] + r*np.sin(th), "--", color="C3", lw=1, zorder=4)
        a.plot(P.goal_rad*np.cos(th), P.y_goal + P.goal_rad*np.sin(th), color="g", lw=1.6)
        a.set(xlim=(-P.corr - 1, P.corr + 1), ylim=(P.y_start - 1, P.y_goal + 1), aspect="equal")
        a.grid(alpha=.25)
        a.set_title(rf"$t = {T_[j]:.1f}$ s,  $\|v\| = {Z[j, 3]:.2f}$ m/s", fontsize=9)
        if i % 2 == 0:
            a.set_ylabel("$q_2$ [m]", fontsize=9)
        if i // 2 == 1:
            a.set_xlabel("$q_1$ [m]", fontsize=9)
    a = fig.add_subplot(gs[2, :])
    a.plot(T_, Z[:, 1], color="C0", lw=1.8, label=r"$q_2(t)$: AS position across the crosswalk")
    a.axhline(P.y_goal - P.goal_rad, color="g", ls="--", lw=1, label=r"edge of $\mathcal{X}_{\rm goal}$")
    for j in idx:
        a.axvline(T_[j], color="0.75", lw=.8, zorder=0)
    a.set_xlabel("time [s]")
    a.set_ylabel("progress $q_2$ [m]", color="C0")
    a.grid(alpha=.3)
    b = a.twinx()
    b.plot(T_, Z[:, 3], color="C3", lw=1, alpha=.85, label=r"$\|v(t)\|$")
    b.set_ylabel(r"$\|v\|$ [m/s]", color="C3")
    b.tick_params(axis="y", colors="C3")
    h1, l1 = a.get_legend_handles_labels()
    h2, l2 = b.get_legend_handles_labels()
    a.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")
    fig.suptitle(f"{NAMES[n]},  $K = {K}$ agents" + ("" if L["reached"] else "  --  goal not reached by T"), fontsize=11)
    fig.savefig(f"fig_timeline_{n}_K{K}.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def outputs(R):
    """Figures on the seed where Truncated OCP completed the largest K, K_T: safety at the largest K
    each method completed, timelines of Truncated OCP at K_T and of Algorithm 1 for every K >= K_T.
    The scaling figure pools all seeds; tables.txt fills the two tables of Section 4 over all seeds."""
    done = lambda s, K, n: (s, K, n) in R and R[s, K, n]["status"] == "ok" and R[s, K, n]["reached"]
    top = lambda s, n: max([K for K in P.Ks if done(s, K, n)], default=0)
    sb = max(P.seeds, key=lambda s: top(s, "trunc"))                    # first seed with the largest K_T
    KT = top(sb, "trunc")
    fig, ax = plt.subplots(1, 3, figsize=(14.4, 3.8))
    for a, n in zip(ax, NAMES):
        if top(sb, n):
            f = np.array(R[sb, top(sb, n), n]["fine"])
            a.plot(f[:, 0], f[:, 1], color="C0", lw=1.4, label=r"$\min_k\|q-q_k\|$")
            a.plot(f[:, 0], f[:, 2], color="C3", lw=1.2, ls="--", label=r"$\Delta_{\mathrm{safe}}\chi_\eta(\|v\|)$")
            a.fill_between(f[:, 0], 0, f[:, 2], color="C3", alpha=.12)
            a.set_title(f"{NAMES[n]}\n$K={top(sb, n)}$", fontsize=9)
            a.set_xlabel("time [s]")
            a.grid(alpha=.3)
    ax[0].set_ylabel("Clearance [m]")
    ax[0].legend(fontsize=8) if ax[0].lines else None
    fig.tight_layout()
    fig.savefig("fig_safety.png", dpi=180)

    fig, a = plt.subplots(figsize=(6.2, 4.2))
    for n in NAMES:                       # median and IQR of solve times pooled over the seeds that completed K
        ok = {K: [s for s in P.seeds if (s, K, n) in R and R[s, K, n]["status"] == "ok"] for K in P.Ks}
        S = {K: np.concatenate([R[s, K, n]["solve"] for s in ok[K]] + [[np.nan]]) for K in P.Ks}
        S = {K: s[np.isfinite(s)] for K, s in S.items() if np.isfinite(s).any()}
        if S:
            a.plot(list(S), [np.median(s) for s in S.values()], "o-", color=COL[n], label=NAMES[n])
            a.fill_between(list(S), [np.percentile(s, 25) for s in S.values()],
                           [np.percentile(s, 75) for s in S.values()], color=COL[n], alpha=.2)
            for K, s in S.items():                                        # completed missions, if not all
                if len(ok[K]) < len(P.seeds):
                    a.annotate(f"{len(ok[K])}/{len(P.seeds)}", (K, np.median(s)), xytext=(5, -11),
                               textcoords="offset points", fontsize=7, color=COL[n])
        for (s, K, m), L in R.items():                                     # the resolution that ended a mission
            if m == n and L["status"] != "ok":
                a.plot(K, L["solve"][-1], "x" if L["status"] == "budget" else "o", ms=8, mew=2, mfc="none",
                       color=COL[n])
    a.plot([], [], "kx", ms=8, mew=2, label="mission abandoned: over budget")
    a.plot([], [], "ko", ms=8, mew=2, mfc="none", label="mission abandoned: infeasible")
    for b, lab, x, ha in ((P.budget, f"maximal budget {P.budget:g} s", .01, "left"),
                          (P.rt_budget, f"real-time budget {1e3*P.rt_budget:g} ms", .99, "right")):
        a.axhline(1e3*b, color="k", ls=":", lw=.9)
        a.text(x, 1.15e3*b, lab, fontsize=8, ha=ha, transform=a.get_yaxis_transform())
    a.set(xscale="log", yscale="log", xlabel="agent count K", ylabel="solve time per iteration [ms]")
    a.set_xticks(P.Ks)
    a.set_xticklabels([K if K not in (75, 125) else "" for K in P.Ks])
    a.minorticks_off()
    a.grid(alpha=.3)
    a.set_ylim(top=30e3*P.budget)                                       # room for the legend
    a.legend(fontsize=7, loc="upper center", ncol=2)
    fig.tight_layout()
    fig.savefig("fig_scaling.png", dpi=180)
    plt.close("all")
    for K in P.Ks:
        if K >= KT and (sb, K, "ours") in R:
            timeline(R[sb, K, "ours"], "ours", K)
    if KT:
        timeline(R[sb, KT, "trunc"], "trunc", KT)

    with open("tables.txt", "w") as f:
        f.write(f"% tab:cost, mean over the seeds where both methods completed, J = dist^2 + int (lam + w|u|^2), lam = {P.lam}, w = {P.w}\n")
        for K in P.Ks:
            S = [s for s in P.seeds if done(s, K, "ours") and done(s, K, "trunc")]
            v = [[R[s, K, n][q] for s in S] for q in ("J", "t_end") for n in ("ours", "trunc")]
            if S:
                f.write(f"${K}$ & " + " & ".join(f"${np.mean(x):.3g}$" for x in v) + rf" \\ % n = {len(S)}, "
                        + "[min, max] " + " ".join(f"[{min(x):.3g}, {max(x):.3g}]" for x in v) + "\n")
        fb = [[100*np.mean(R[s, K, "ours"]["fallback"] or [0]) for s in P.seeds if (s, K, "ours") in R]
              or [np.nan] for K in P.Ks]
        f.write("% tab:fallback, mean over seeds, then [min, max]\n" + r"fallback [\%] & "
                + " & ".join(f"${np.mean(x):.0f}$" for x in fb) + r" \\" + "\n% "
                + " ".join(f"[{min(x):.0f}, {max(x):.0f}]" for x in fb) + "\n")
        for n in NAMES:
            Ls = [L for (s, K, m), L in R.items() if m == n]
            f.write(f"% {n}: completed missions per K " + " ".join(
                f"{K}:{sum(done(s, K, n) for s in P.seeds)}/{len(P.seeds)}" for K in P.Ks)
                + f"; violations > {P.tol} m: {sum(L['viol'] for L in Ls)}; worst margin {np.nanmin([L['margin'] for L in Ls]):+.3f} m\n")
        f.write(f"% figures: seed {sb}, Truncated OCP largest completed K = {KT}\n")
    print(open("tables.txt").read())


def main():
    if sys.argv[1:] == ["plots"]:
        return outputs(pickle.load(open("results.pkl", "rb")))
    for K in P.Ks:
        print(f"K={K:3d}: M_crowd {(K - 1)*MG:6.2f}  C_agent {P.A_s + (K - 1)*MG:6.2f}  "
              f"M_dmax {M_dmax(K):5.2f}  (13) {'OK' if 0 < M_dmax(K) <= P.eta else 'FAILS'}")
    ours, base, R = build_ours(), {}, {}
    for seed in P.seeds:
        dead = dict.fromkeys(NAMES, False)
        for K in P.Ks:
            for n, m in (("ours", 0), ("full", K), ("trunc", int(np.ceil(np.log(K))))):
                if dead[n]:
                    continue
                if m and m not in base:
                    base[m] = build_base(m)
                R[seed, K, n] = L = run_ours(K, seed, ours) if n == "ours" else run_base(K, seed, m, base[m])
                report(seed, K, n, L)
                dead[n] = L["status"] != "ok"
    pickle.dump(R, open("results.pkl", "wb"))
    outputs(R)


if __name__ == "__main__":
    main()









