#!/usr/bin/env python3
"""build_closure.py -- offline exact parametric closure, eq. (9), with D = D_agent.

D = D_agent only bounds |v_k| < v_agent, so every neighbour of the closest agent may sit anywhere
with any admissible velocity: co F_crowd(x_a) = (K-1) co G(v_a), G(v_a) being the forces that ONE
neighbour can exert.  D is rotation invariant, so co G(v_a) = R(v_a) co G(|v_a| e_1), a body that is
anisotropic (D bounds absolute velocities, f_crowd sees relative ones).  Eq. (9) is positively
homogeneous and rotation equivariant, so f_hat(x_a, R(v_a) p) = (K-1) R(v_a) f_hat_G(|v_a|, p).
We tabulate f_hat_G(s, p) for s on a uniform grid of step h_s (interpolated linearly online) and
p on the grid of the closed unit ball on which constraint S is imposed.
Run: python build_closure.py
"""


import numpy as np
from scipy.spatial import ConvexHull
from common import P, pair_force


def body(s, nr=40, na=72, nv=7, nva=36):
    """CCW vertices of co G(s e_1): forces of one neighbour at any relative position (within the
    cut-off) with any velocity |v_j| <= v_agent, plus 0 (neighbour out of range)."""
    r, a = np.meshgrid(np.linspace(0, P.R_c, nr + 1)[1:], np.linspace(0, 2*np.pi, na, endpoint=False))
    rv, av = np.meshgrid(np.linspace(0, P.v_agent, nv), np.linspace(0, 2*np.pi, nva, endpoint=False))
    x = np.c_[(r*np.cos(a)).ravel(), (r*np.sin(a)).ravel()][:, None]                  # q_j - q_a
    v = np.array([s, 0.0]) - np.c_[(rv*np.cos(av)).ravel(), (rv*np.sin(av)).ravel()][None]  # v_a - v_j
    fx, fy = pair_force(x[..., 0], x[..., 1], v[..., 0], v[..., 1])
    F = np.r_[np.c_[fx.ravel(), fy.ravel()], [[0.0, 0.0]]]
    return F[ConvexHull(F).vertices]


def proj(y, V):
    """Nearest point of the convex polygon V (CCW vertices) to y, and its distance (0 inside)."""
    E = np.roll(V, -1, 0) - V
    if np.all(E[:, 0]*(y[1] - V[:, 1]) - E[:, 1]*(y[0] - V[:, 0]) >= 0):
        return y, 0.0
    C = V + np.clip(((y - V)*E).sum(1)/(E*E).sum(1), 0, 1)[:, None]*E
    i = np.argmin(np.linalg.norm(C - y, axis=1))
    return C[i], np.linalg.norm(C[i] - y)


def clip(V, C):
    """Sutherland-Hodgman: convex polygon V clipped by the convex CCW polygon C."""
    for a, b in zip(C, np.roll(C, -1, 0)):
        s = (b[0] - a[0])*(V[:, 1] - a[1]) - (b[1] - a[1])*(V[:, 0] - a[0])
        out = []
        for i in range(len(V)):
            j = (i + 1) % len(V)
            if s[i] >= 0:
                out.append(V[i])
            if s[i]*s[j] < 0:
                out.append(V[i] + s[i]/(s[i] - s[j])*(V[j] - V[i]))
        if len(out) < 3:
            return np.zeros((0, 2))
        V = np.array(out)
    return V


def steiner(V):
    """Steiner point of a convex polygon: vertices weighted by their exterior angles."""
    V = V[np.linalg.norm(V - np.roll(V, 1, 0), axis=1) > 1e-12]
    E = np.roll(V, -1, 0) - V
    Ei = np.roll(E, 1, 0)                                          # edge entering each vertex
    phi = np.arctan2(Ei[:, 0]*E[:, 1] - Ei[:, 1]*E[:, 0], (Ei*E).sum(1))
    return phi @ V/phi.sum()


def f_hat(V, M, p, m=64):
    """Eq. (9): Steiner point of co G cap B(M p, 2 dist(M p, co G)), the ball as an m-gon."""
    y = M*p
    c, d = proj(y, V)
    if d == 0:
        return y
    b = 2*np.pi*np.arange(m)/m
    W = clip(V, y + 2*d*np.c_[np.cos(b), np.sin(b)])
    return steiner(W) if len(W) >= 3 else c


def p_grid():
    """Grid of the closed unit ball, spacing ~h_p, boundary circle included."""
    pts = [[0.0, 0.0]]
    for r in np.linspace(0, 1, int(np.ceil(1/P.h_p)) + 1)[1:]:
        n = max(3, int(round(2*np.pi*r/P.h_p)))
        pts += [[r*np.cos(2*np.pi*i/n), r*np.sin(2*np.pi*i/n)] for i in range(n)]
    return np.array(pts)


if __name__ == "__main__":
    SG, PG = np.linspace(0, P.v_agent, int(round(P.v_agent/P.h_s)) + 1), p_grid()
    B = [body(s) for s in SG]
    M = np.array([np.linalg.norm(b, axis=1).max() for b in B])     # M_crowd / (K-1), per |v_a|
    T = np.array([[f_hat(b, m, p) for p in PG] for b, m in zip(B, M)])
    np.savez("closure.npz", s_grid=SG, p_grid=PG, T=T, M=M)
    print(f"{len(SG)} speeds x {len(PG)} parameters; one neighbour: max force {M.max():.4f} m/s^2")
    print(f"Steiner point of co G: {steiner(B[0]).round(4)} at |v_a| = 0, "
          f"{steiner(B[-1]).round(4)} at |v_a| = {P.v_agent}")
    print("   K   M_crowd  C_agent  M_dmax   (13)")
    for K in P.Ks:
        C = P.A_s + (K - 1)*M.max()
        Md = P.rho*P.Delta_safe - P.v_agent - C*P.delta_max
        print(f"{K:4d} {(K - 1)*M.max():8.2f} {C:8.2f} {Md:7.2f}   {'OK' if 0 < Md <= P.eta else 'FAILS'}")








