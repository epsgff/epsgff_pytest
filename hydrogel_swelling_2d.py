"""
2D simulation of a square hydrogel swelling when one side touches water.

Model (dimensionless, stress in units of kT/v, v = volume of a water molecule)
---------------------------------------------------------------------------
1. Flory-Rehner theory
   The chemical potential of water in a freely swelling gel is
       mu/kT = ln(1 - phi) + phi + chi*phi^2 + Nv*(phi^(1/3) - phi/2)
   with phi = 1/J the polymer volume fraction, chi the Flory-Huggins
   interaction parameter and Nv the crosslink density. mu = 0 (gel in
   equilibrium with pure water) gives the equilibrium swelling ratio J_eq.

2. Water uptake (absorption)
   s(x, y, t) in [0, 1] is the normalized water content. Water enters from
   the left edge only (s = 1 there, since that edge is in contact with
   water). The other edges are impermeable. s follows a diffusion equation
   on the reference (dry) mesh:  ds/dt = D * laplace(s).
   The local free-swelling volume ratio is J_s = 1 + s*(J_eq - 1), so the
   local isotropic swelling stretch is lambda_s = J_s^(1/3).

3. Mechanics (large deformation, finite elements)
   The deformation gradient is split as F = F_e * F_s with F_s = lambda_s*I.
   Swelling is non-uniform (wet side swells, dry side does not), so the
   elastic part F_e is needed to keep the body compatible, which produces
   internal stress. F_e is governed by a compressible neo-Hookean energy
   whose shear modulus follows Flory's rubber elasticity, G = Nv*phi^(1/3).
   Boundary: the right edge and the right halves of the top and bottom
   edges are clamped (u = 0, green line in the plots); the rest is free.
   Each frame solves for mechanical equilibrium by minimizing the total
   elastic energy with Newton's method.

Output: a heat-map animation of the deformed mesh (left: water content,
right: internal stress) plus a snapshot figure.

Usage:
    python hydrogel_swelling_2d.py                 # von Mises stress
    python hydrogel_swelling_2d.py --stress hydro  # hydrostatic stress
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.optimize import brentq
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

KT_OVER_V = 1.38e8  # Pa, kT/v at 300 K with v = 3e-29 m^3


# --------------------------------------------------------------------------
# Flory-Rehner theory
# --------------------------------------------------------------------------
def flory_chemical_potential(phi, chi, Nv):
    """Water chemical potential (in kT) of a freely swollen gel."""
    return np.log(1 - phi) + phi + chi * phi**2 + Nv * (phi ** (1 / 3) - phi / 2)


def equilibrium_swelling_ratio(chi, Nv):
    """Solve mu(phi) = 0 and return J_eq = 1/phi_eq."""
    phi_eq = brentq(flory_chemical_potential, 1e-8, 1 - 1e-12, args=(chi, Nv))
    return 1.0 / phi_eq


# --------------------------------------------------------------------------
# Mesh
# --------------------------------------------------------------------------
def make_square_mesh(n, L):
    """Structured triangular mesh of an L x L square with n x n cells."""
    xs = np.linspace(0.0, L, n + 1)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    nodes = np.column_stack([X.ravel(), Y.ravel()])

    def idx(i, j):
        return i * (n + 1) + j

    tris = []
    for i in range(n):
        for j in range(n):
            a, b, c, d = idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)
            # alternate diagonals to avoid a directional bias
            if (i + j) % 2 == 0:
                tris += [[a, b, c], [a, c, d]]
            else:
                tris += [[a, b, d], [b, c, d]]
    return nodes, np.array(tris)


# --------------------------------------------------------------------------
# Water diffusion (linear P1 finite elements, implicit Euler)
# --------------------------------------------------------------------------
class Diffusion:
    def __init__(self, nodes, tris, D, wet_nodes):
        n_nodes = len(nodes)
        rows, cols, kv, mv = [], [], [], []
        for t in tris:
            p = nodes[t]
            Dm = np.column_stack([p[1] - p[0], p[2] - p[0]])
            area = 0.5 * abs(np.linalg.det(Dm))
            # gradients of the three shape functions
            G = np.linalg.inv(Dm).T @ np.array([[-1.0, 1.0, 0.0], [-1.0, 0.0, 1.0]])
            Ke = area * G.T @ G
            Me = area / 12.0 * (np.ones((3, 3)) + np.eye(3))
            for a in range(3):
                for b in range(3):
                    rows.append(t[a])
                    cols.append(t[b])
                    kv.append(Ke[a, b])
                    mv.append(Me[a, b])
        shape = (n_nodes, n_nodes)
        self.K = coo_matrix((kv, (rows, cols)), shape=shape).tocsr()
        self.M = coo_matrix((mv, (rows, cols)), shape=shape).tocsr()
        self.D = D
        self.wet = wet_nodes
        self.free = np.setdiff1d(np.arange(n_nodes), wet_nodes)

    def step(self, s, dt):
        A = (self.M + dt * self.D * self.K).tocsr()
        rhs = self.M @ s
        s_new = s.copy()
        s_new[self.wet] = 1.0  # left edge is in contact with water
        rhs -= A[:, self.wet] @ s_new[self.wet]
        f = self.free
        s_new[f] = spsolve(A[f][:, f].tocsc(), rhs[f])
        return np.clip(s_new, 0.0, 1.0)


# --------------------------------------------------------------------------
# Mechanics: neo-Hookean with swelling eigen-stretch
# --------------------------------------------------------------------------
class Mechanics:
    def __init__(self, nodes, tris, Nv, bulk_ratio, fixed_dofs):
        self.X = nodes
        self.tris = tris
        self.Nv = Nv
        self.bulk_ratio = bulk_ratio
        p0, p1, p2 = (nodes[tris[:, k]] for k in range(3))
        Dm = np.stack([p1 - p0, p2 - p0], axis=-1)  # (ne, 2, 2)
        self.Dm_inv = np.linalg.inv(Dm)
        self.A0 = 0.5 * np.linalg.det(Dm)
        self.fixed = fixed_dofs
        self.free = np.setdiff1d(np.arange(nodes.size), fixed_dofs)

    def _kinematics(self, x, lam):
        t = self.tris
        Ds = np.stack([x[t[:, 1]] - x[t[:, 0]], x[t[:, 2]] - x[t[:, 0]]], axis=-1)
        F = Ds @ self.Dm_inv
        Fe = F / lam[:, None, None]
        return Fe

    def _moduli(self, lam):
        # Flory rubber elasticity: G = NkT * phi^(1/3), phi = 1/J_s = lam^-3
        mu = self.Nv / lam
        return mu, self.bulk_ratio * mu

    def energy_and_grad(self, u_free, u_full, lam):
        u = u_full.copy()
        u[self.free] = u_free
        x = self.X + u.reshape(-1, 2)
        Fe = self._kinematics(x, lam)
        J = np.linalg.det(Fe)
        if np.any(J <= 1e-8):
            return 1e30, np.zeros_like(u_free)
        mu, lmb = self._moduli(lam)
        lnJ = np.log(J)
        I1 = np.einsum("eij,eij->e", Fe, Fe)
        W = 0.5 * mu * (I1 - 2.0 - 2.0 * lnJ) + 0.5 * lmb * lnJ**2
        # energy per swollen (stress-free) area = A0 * lam^2
        vol = self.A0 * lam**2
        E = np.sum(vol * W)

        FinvT = np.linalg.inv(Fe).transpose(0, 2, 1)
        P = mu[:, None, None] * (Fe - FinvT) + (lmb * lnJ)[:, None, None] * FinvT
        # dE/dF = vol * P / lam ; dE/dDs = dE/dF * Dm_inv^T
        H = (vol / lam)[:, None, None] * P @ self.Dm_inv.transpose(0, 2, 1)
        f = np.zeros_like(x)
        t = self.tris
        np.add.at(f, t[:, 1], H[:, :, 0])
        np.add.at(f, t[:, 2], H[:, :, 1])
        np.add.at(f, t[:, 0], -H[:, :, 0] - H[:, :, 1])
        return E, f.ravel()[self.free]

    def hessian(self, u, lam):
        """Consistent tangent stiffness (sparse), restricted to free dofs."""
        x = self.X + u.reshape(-1, 2)
        Fe = self._kinematics(x, lam)
        J = np.linalg.det(Fe)
        mu, lmb = self._moduli(lam)
        Fi = np.linalg.inv(Fe)  # Fi[e, J, i] = (Fe^-1)_Ji
        I2 = np.eye(2)
        # A_iJkL = dP_iJ / dFe_kL for the neo-Hookean model
        A = (mu[:, None, None, None, None] * np.einsum("ik,JL->iJkL", I2, I2)[None]
             + (mu - lmb * np.log(J))[:, None, None, None, None] * np.einsum("eJk,eLi->eiJkL", Fi, Fi)
             + lmb[:, None, None, None, None] * np.einsum("eJi,eLk->eiJkL", Fi, Fi))
        vol = self.A0 * lam**2
        B = self.Dm_inv / lam[:, None, None]  # dFe/dDs
        H = vol[:, None, None, None, None] * np.einsum("eiJkL,eMJ,eNL->eiMkN", A, B, B)
        T = np.array([[-1.0, 1.0, 0.0], [-1.0, 0.0, 1.0]])  # Ds columns from nodes
        Ke = np.einsum("Ma,Nb,eiMkN->eaibk", T, T, H).reshape(-1, 6, 6)
        dofs = np.stack([2 * self.tris, 2 * self.tris + 1], axis=-1).reshape(-1, 6)
        rows = np.repeat(dofs, 6, axis=1).ravel()
        cols = np.tile(dofs, (1, 6)).ravel()
        K = coo_matrix((Ke.ravel(), (rows, cols)), shape=(self.X.size,) * 2).tocsr()
        f = self.free
        return K[f][:, f].tocsc()

    def solve(self, u, lam, tol=1e-9, max_iter=50):
        """Newton-Raphson with backtracking line search on the total energy."""
        u = u.copy()
        f = self.free
        scale = self.Nv * self.A0.sum()
        for _ in range(max_iter):
            E, g = self.energy_and_grad(u[f], u, lam)
            if np.abs(g).max() < tol * scale:
                break
            du = spsolve(self.hessian(u, lam), -g)
            if g @ du >= 0:  # not a descent direction -> gradient step
                du = -g / np.abs(g).max() * 1e-2
            step = 1.0
            while step > 1e-8:
                E_new, _ = self.energy_and_grad(u[f] + step * du, u, lam)
                if E_new <= E + 1e-4 * step * (g @ du):
                    break
                step *= 0.5
            u[f] += step * du
        return u

    def cauchy_stress(self, u, lam):
        """Cauchy stress per element: sigma = J^-1 P Fe^T."""
        x = self.X + u.reshape(-1, 2)
        Fe = self._kinematics(x, lam)
        J = np.linalg.det(Fe)
        mu, lmb = self._moduli(lam)
        b = Fe @ Fe.transpose(0, 2, 1)
        I = np.eye(2)[None]
        sigma = (mu[:, None, None] * (b - I) + (lmb * np.log(J))[:, None, None] * I) / J[:, None, None]
        return sigma


def stress_measure(sigma, kind):
    sxx, syy, sxy = sigma[:, 0, 0], sigma[:, 1, 1], sigma[:, 0, 1]
    if kind == "hydro":
        return 0.5 * (sxx + syy)
    return np.sqrt(sxx**2 - sxx * syy + syy**2 + 3 * sxy**2)  # von Mises


def element_to_node(values, tris, n_nodes):
    acc = np.zeros(n_nodes)
    cnt = np.zeros(n_nodes)
    for k in range(3):
        np.add.at(acc, tris[:, k], values)
        np.add.at(cnt, tris[:, k], 1)
    return acc / cnt


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=24, help="cells per side")
    ap.add_argument("--chi", type=float, default=0.6, help="Flory-Huggins parameter")
    ap.add_argument("--Nv", type=float, default=0.001, help="crosslink density N*v")
    ap.add_argument("--D", type=float, default=1.0, help="water diffusivity")
    ap.add_argument("--t_end", type=float, default=1.5, help="simulated time (L^2/D)")
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--stress", choices=["vm", "hydro"], default="vm")
    ap.add_argument("--out", default="hydrogel_output")
    args = ap.parse_args()

    L = 1.0
    J_eq = equilibrium_swelling_ratio(args.chi, args.Nv)
    print(f"Flory-Rehner: chi={args.chi}, Nv={args.Nv} -> J_eq={J_eq:.3f}, "
          f"linear stretch={J_eq ** (1 / 3):.3f}")

    nodes, tris = make_square_mesh(args.n, L)
    n_nodes = len(nodes)
    wet = np.where(np.isclose(nodes[:, 0], 0.0))[0]

    # clamped (u = 0) boundary: the whole right edge, plus the right half
    # (x >= L/2, from the midpoint to the corner) of the top and bottom edges
    on_right = np.isclose(nodes[:, 0], L)
    on_top_bottom = np.isclose(nodes[:, 1], 0.0) | np.isclose(nodes[:, 1], L)
    clamped = np.where(on_right | (on_top_bottom & (nodes[:, 0] >= 0.5 * L - 1e-12)))[0]
    fixed = np.sort(np.concatenate([2 * clamped, 2 * clamped + 1]))

    diff = Diffusion(nodes, tris, args.D, wet)
    mech = Mechanics(nodes, tris, args.Nv, bulk_ratio=10.0, fixed_dofs=fixed)

    s = np.zeros(n_nodes)
    u = np.zeros(nodes.size)
    times = args.t_end * (np.arange(args.frames + 1) / args.frames) ** 2  # finer early
    substeps = 5
    history = []
    for k, t in enumerate(times):
        if k > 0:
            dt = (t - times[k - 1]) / substeps
            for _ in range(substeps):
                s = diff.step(s, dt)
        s_elem = s[tris].mean(axis=1)
        J_s = 1.0 + s_elem * (J_eq - 1.0)
        lam = J_s ** (1.0 / 3.0)
        u = mech.solve(u, lam)
        sig = mech.cauchy_stress(u, lam)
        st = stress_measure(sig, args.stress) * KT_OVER_V / 1e3  # kPa
        area = mech.A0.sum()
        uptake = np.sum(mech.A0 * s_elem * (J_eq - 1.0)) / area  # water volume / dry volume
        history.append(dict(t=t, s=s.copy(), u=u.copy(), stress=element_to_node(st, tris, n_nodes),
                            uptake=uptake))
        print(f"frame {k:3d}  t={t:.4f}  water uptake={uptake:.3f}  "
              f"max stress={np.abs(st).max():8.1f} kPa")

    # ----------------------------- plotting -----------------------------
    os.makedirs(args.out, exist_ok=True)
    all_x = np.concatenate([nodes + h["u"].reshape(-1, 2) for h in history])
    pad = 0.08
    xlim = (all_x[:, 0].min() - pad, all_x[:, 0].max() + pad)
    ylim = (all_x[:, 1].min() - pad, all_x[:, 1].max() + pad)
    all_st = np.concatenate([h["stress"] for h in history])
    # colour range from the 99th percentile so a single early peak at the
    # thin wet layer does not wash out the rest of the animation
    if args.stress == "hydro":
        vmax = np.percentile(np.abs(all_st), 99)
        st_norm, st_cmap = dict(vmin=-vmax, vmax=vmax), "coolwarm"
        st_label = "hydrostatic stress (kPa)  (+ tension / - compression)"
    else:
        st_norm, st_cmap = dict(vmin=0.0, vmax=np.percentile(all_st, 99)), "inferno"
        st_label = "von Mises stress (kPa)"

    # clamped boundary is fixed, so its path is the same in every frame
    bottom = clamped[np.isclose(nodes[clamped, 1], 0.0)]
    top = clamped[np.isclose(nodes[clamped, 1], L)]
    right = clamped[on_right[clamped]]
    clamp_path = np.vstack([nodes[bottom[np.argsort(nodes[bottom, 0])]],
                            nodes[right[np.argsort(nodes[right, 1])]],
                            nodes[top[np.argsort(-nodes[top, 0])]]])

    def draw(axes, h, cbars=None):
        x = nodes + h["u"].reshape(-1, 2)
        tri = mtri.Triangulation(x[:, 0], x[:, 1], tris)
        ax_w, ax_s = axes
        for ax in axes:
            ax.clear()
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_aspect("equal")
            # water reservoir on the left side
            xl = x[wet]
            xl = xl[np.argsort(xl[:, 1])]
            ax.fill_betweenx(xl[:, 1], xlim[0], xl[:, 0], color="#5aa9e6", alpha=0.35, lw=0)
        pw = ax_w.tripcolor(tri, h["s"], shading="gouraud", cmap="Blues", vmin=0, vmax=1)
        ps = ax_s.tripcolor(tri, h["stress"], shading="gouraud", cmap=st_cmap, **st_norm)
        for ax in axes:
            ax.triplot(tri, color="k", lw=0.3, alpha=0.5)
            # clamped boundary: bottom half-edge -> right edge -> top half-edge
            ax.plot(*clamp_path.T, color="#2e7d32", lw=4, solid_capstyle="butt", zorder=5)
        ax_w.set_title(f"water content   t = {h['t']:.3f} L²/D\nuptake = {h['uptake']:.2f} × dry volume")
        ax_s.set_title(f"internal stress on deformed mesh\nmax = {np.abs(h['stress']).max():.1f} kPa")
        return pw, ps

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), layout="constrained")
    pw, ps = draw(axes, history[0])
    fig.colorbar(pw, ax=axes[0], label="normalized water content s")
    fig.colorbar(ps, ax=axes[1], label=st_label)
    fig.suptitle(f"Square hydrogel swelling from one side (Flory-Rehner, χ={args.chi}, Nv={args.Nv})")

    anim = FuncAnimation(fig, lambda k: draw(axes, history[k]), frames=len(history))
    gif_path = os.path.join(args.out, f"hydrogel_swelling_{args.stress}.gif")
    anim.save(gif_path, writer=PillowWriter(fps=12), dpi=80)
    plt.close(fig)
    print("saved", gif_path)

    # snapshot grid
    nf = len(history) - 1
    picks = [nf // 20, nf // 6, nf // 3, nf]
    fig, axes = plt.subplots(2, 4, figsize=(18, 8.5), layout="constrained")
    for col, k in enumerate(picks):
        pw, ps = draw(axes[:, col], history[k])
    fig.colorbar(pw, ax=axes[0, :], label="water content s")
    fig.colorbar(ps, ax=axes[1, :], label=st_label)
    png_path = os.path.join(args.out, f"hydrogel_snapshots_{args.stress}.png")
    fig.savefig(png_path, dpi=110)
    plt.close(fig)
    print("saved", png_path)


if __name__ == "__main__":
    main()
