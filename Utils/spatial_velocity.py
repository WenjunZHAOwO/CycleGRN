import scanpy as sc
import anndata as ad
from cycleGRN import find_Lie_derivative

from sklearn.neighbors import NearestNeighbors
import numpy as np
from scipy.sparse import csr_matrix


import numpy as np
from scipy.sparse import csr_matrix, eye
from scipy.sparse.linalg import spsolve

def find_diffusion_time(T, terminal_mask, eps=1e-12):
    """
    Compute expected hitting (diffusion) time to terminal states.

    Parameters
    ----------
    T : scipy.sparse.csr_matrix
        Row-stochastic transition matrix.
    terminal_mask : array-like (bool)
        Boolean mask indicating terminal states.
    eps : float
        Numerical tolerance.

    Returns
    -------
    tau : np.ndarray, shape (n,)
        Expected diffusion / absorption time for each state.
        Terminal states have tau = 0.
    """
    if not isinstance(T, csr_matrix):
        T = csr_matrix(T)

    n = T.shape[0]
    terminal_mask = np.asarray(terminal_mask, dtype=bool)

    # Identify transient states
    transient_mask = ~terminal_mask
    idx_trans = np.where(transient_mask)[0]


    if len(idx_trans) == 0:
        if verbose:
            print("No transient states: tau = 0 everywhere.")
        return np.zeros(n)

    # Submatrix on transient states
    Q = T[transient_mask][:, transient_mask]

    # Diagnostics: row sums
    row_sums = np.asarray(Q.sum(axis=1)).ravel()
    if verbose:
        print(
            "Transient row-sum min/max:",
            row_sums.min(),
            row_sums.max()
        )


    # Fundamental matrix equation:
    # tau = 1 + Q tau  ->  (I - Q) tau = 1
    I = eye(Q.shape[0], format="csr")
    b = np.ones(Q.shape[0])

    # Solve linear system
    tau_trans = spsolve(I - Q, b)

    # Assemble full vector
    tau = np.zeros(n)
    tau[transient_mask] = tau_trans
    tau[terminal_mask] = np.min(tau_trans)
    tau = (tau - tau[0] ) / (np.max(tau) - tau[0])
   
    
    return tau


# def project_velocity_spatial(adata, T, k=50, basis='spatial'):
#     X = adata.obsm['spatial'][:, :2]
#     # Tsub = T.T
    
#     k = min(k, X.shape[0]-1)
#     nn = NearestNeighbors(n_neighbors=k).fit(X)
#     nbrs = nn.kneighbors(return_distance=False)
    
#     rows = np.repeat(np.arange(X.shape[0]), k)
#     cols = nbrs.ravel()
#     mask = csr_matrix((np.ones_like(cols), (rows, cols)), shape=T.shape)
    
#     Tsub = T.multiply(mask)
    
#     rs = np.asarray(Tsub.sum(axis=1)).ravel()
#     rs[rs == 0] = 1.0
#     Tsub = Tsub.multiply(1.0 / rs[:, None])
#     if basis == 'spatial':
#         V_mask = find_Lie_derivative(adata.obsm[basis].T, Tsub, stimulation=False)
        
#         rs = np.asarray(mask.sum(axis=1)).ravel()
#         rs[rs == 0] = 1.0
#         mask = mask.multiply(1.0 / rs[:, None])
#         V_gated = find_Lie_derivative(adata.obsm[basis].T, mask, stimulation=False)
#     else:
#         V_mask = find_Lie_derivative(adata.X.todense().T, Tsub, stimulation=False)
        
#         rs = np.asarray(mask.sum(axis=1)).ravel()
#         rs[rs == 0] = 1.0
#         mask = mask.multiply(1.0 / rs[:, None])
#         V_gated = find_Lie_derivative(adata.X.todense().T, mask, stimulation=False)
#     V_overall = V_mask - V_gated

#     return V_overall, V_mask, V_gated

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

def _row_sum(A):
    return np.asarray(A.sum(axis=1)).ravel()

def _safe_row_normalize(A, eps=1e-12):
    """Row-normalize sparse matrix A; leaves zero rows as zeros."""
    rs = _row_sum(A)
    inv = np.zeros_like(rs, dtype=float)
    inv[rs > eps] = 1.0 / rs[rs > eps]
    return A.multiply(inv[:, None])

def _knn_mask(X, k=50, include_self=False, symmetrize=False):
    """
    Build a directed (or symmetrized) kNN adjacency mask M (n x n),
    with ones on k neighbor entries per row (unless near-degenerate cases).
    """
    n = X.shape[0]
    if n <= 1:
        raise ValueError("Need at least 2 points to build kNN mask.")
    k_eff = min(k, n - 1)

    # If we want to exclude self (recommended), request k+1 and drop self index.
    k_query = k_eff + (0 if include_self else 1)
    nn = NearestNeighbors(n_neighbors=k_query, algorithm="auto").fit(X)
    nbrs = nn.kneighbors(return_distance=False)

    if include_self:
        nbrs_use = nbrs[:, :k_eff]
    else:
        # Typically nbrs[:, 0] is self; drop it. If not, still safe-ish.
        nbrs_use = nbrs[:, 1:k_eff + 1]

    rows = np.repeat(np.arange(n), nbrs_use.shape[1])
    cols = nbrs_use.ravel()
    data = np.ones_like(cols, dtype=float)
    M = csr_matrix((data, (rows, cols)), shape=(n, n))

    if symmetrize:
        # Mutual neighborhood (undirected) mask
        M = ((M + M.T) > 0).astype(float).tocsr()

    return M

def project_velocity_spatial(
    adata,
    T,
    k=50,
    basis="spatial",
    symmetrize_mask=False,
    include_self=False,
    eps=1e-12,
):
    """
    Spatially gate a transition matrix T using kNN neighborhoods in adata.obsm['spatial'].

    Returns:
        V_overall: dynamics-specific drift (contrastive)
        V_dyn:     drift from gated dynamics
        V_base:    baseline drift from matched-mass uniform mixing on the same support
    """

    # --- 1) Build spatial mask M ---
    if "spatial" not in adata.obsm:
        raise KeyError("adata.obsm['spatial'] not found.")
    X_sp = adata.obsm["spatial"][:, :2]
    M = _knn_mask(X_sp, k=k, include_self=include_self, symmetrize=symmetrize_mask)

    # --- 2) Gate T to neighbors ---
    if T.shape != M.shape:
        raise ValueError(f"T shape {T.shape} must match mask shape {M.shape}.")

    Tg = T.multiply(M)  # support restricted to spatial neighbors

    # Row mass after gating (before normalization)
    mass = _row_sum(Tg)

    # --- 3) Construct baseline with the SAME support and SAME row mass ---
    # Baseline = uniform over neighbors, scaled to have row sums equal to mass.
    # If a row has zero mass (no gated transition), baseline row will be zero too.
    deg = _row_sum(M)
    invdeg = np.zeros_like(deg, dtype=float)
    invdeg[deg > eps] = 1.0 / deg[deg > eps]
    U = M.multiply(invdeg[:, None])                 # row-stochastic uniform over neighbors
    B = U.multiply(mass[:, None])                   # match row mass to Tg

    # --- 4) Convert both to row-stochastic operators if your Lie derivative expects that ---
    # If find_Lie_derivative expects row-stochastic, normalize Tg and B.
    Tg_stoch = _safe_row_normalize(Tg, eps=eps)
    B_stoch  = _safe_row_normalize(B,  eps=eps)

    # --- 5) Choose coordinate basis matrix for Lie derivative ---
    if basis == "spatial":
        X = adata.obsm["spatial"][:, :2]            # (n,2)
    elif basis in adata.obsm:
        X = adata.obsm[basis]
    else:
        # Use expression matrix without densifying
        X = adata.X
        if hasattr(X, "toarray"):
            # find_Lie_derivative may want dense; if so, consider a controlled conversion
            X = X.toarray()

    # The original code passes X.T into find_Lie_derivative; keep that convention:
    # X should be (n,d) so X.T is (d,n)
    X_T = X.T

    # --- 6) Compute contrastive drift ---
    V_dyn  = find_Lie_derivative(X_T, Tg_stoch, stimulation=False)
    V_base = find_Lie_derivative(X_T, B_stoch,  stimulation=False)
    V_overall = V_dyn - V_base

    return V_overall, V_dyn, V_base





    
def velocity_streamplot_spatial(
    x, y, vx, vy,
    gridsize=120,
    method="linear",
    smooth_sigma=1.5,
    density=1.1,
    min_count=1,
    radius=None,
    ax=None,
    cmap="plasma",
    linewidth_range=(1.0, 3.5),   # scale linewidth by speed
    add_colorbar=False,
    **stream_kwargs
):
    """
    Streamplot from scattered vectors with:
      (1) density-based support masking
      (2) NaN-safe Gaussian smoothing
      (3) coloring + linewidth by velocity magnitude
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.interpolate import griddata
    from scipy.ndimage import gaussian_filter
    from scipy.spatial import cKDTree
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    vx = np.asarray(vx).ravel()
    vy = np.asarray(vy).ravel()

    if ax is None:
        _, ax = plt.subplots()

    # -------------------------
    # Grid
    # -------------------------
    xi = np.linspace(x.min(), x.max(), gridsize)
    yi = np.linspace(y.min(), y.max(), gridsize)
    X, Y = np.meshgrid(xi, yi)

    # -------------------------
    # Interpolate
    # -------------------------
    U = griddata((x, y), vx, (X, Y), method=method)
    V = griddata((x, y), vy, (X, Y), method=method)

    # -------------------------
    # Support mask
    # -------------------------
    tree = cKDTree(np.c_[x, y])
    if radius is None:
        dx = (x.max() - x.min()) / max(gridsize - 1, 1)
        dy = (y.max() - y.min()) / max(gridsize - 1, 1)
        radius = 3.0 * max(dx, dy)

    counts = tree.query_ball_point(
        np.c_[X.ravel(), Y.ravel()],
        r=radius,
        return_length=True
    ).reshape(X.shape)

    support = counts >= min_count
    valid = support & np.isfinite(U) & np.isfinite(V)
    U = np.where(valid, U, np.nan)
    V = np.where(valid, V, np.nan)

    # -------------------------
    # NaN-safe smoothing
    # -------------------------
    if smooth_sigma and smooth_sigma > 0:
        W = np.isfinite(U).astype(float)
        U0 = np.nan_to_num(U, nan=0.0)
        V0 = np.nan_to_num(V, nan=0.0)

        UW = gaussian_filter(U0 * W, smooth_sigma)
        VW = gaussian_filter(V0 * W, smooth_sigma)
        WW = gaussian_filter(W, smooth_sigma)

        eps = 1e-12
        U = UW / (WW + eps)
        V = VW / (WW + eps)

        U[~support] = np.nan
        V[~support] = np.nan

    # -------------------------
    # Velocity magnitude (for color / linewidth)
    # -------------------------
    speed = np.sqrt(U**2 + V**2)

    # Robust normalization (ignore extreme outliers)
    vmax = np.nanpercentile(speed, 95)
    norm = Normalize(vmin=0.0, vmax=vmax)

    # Linewidth scaling
    if linewidth_range is not None:
        lw_min, lw_max = linewidth_range
        linewidth = lw_min + (lw_max - lw_min) * norm(speed)
    else:
        linewidth = 1.0

    # -------------------------
    # Streamplot
    # -------------------------
    strm = ax.streamplot(
        X, Y, U, V,
        color=speed,
        cmap=cmap,
        norm=norm,
        linewidth=linewidth,
        density=density,
        arrowsize=1.3,
        **stream_kwargs
    )

    # -------------------------
    # Optional colorbar
    # -------------------------
    if add_colorbar:
        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02, label="Velocity magnitude")

    return ax

def orient_flow_by_cell_cycle_phase(
    adata,
    T,
    phase_key="phase",              # e.g. "phase" or "phase_cc" or "cell_cycle_phase"
    phase_order=("G1", "S", "G2M"),  # assumed forward order
    use_empirical_weights=False,     # False = weight phases equally
    flip_strategy="transpose",       # "transpose" or "negate_velocity"
    verbose=True,
):
    """
    Given AnnData with discrete cell-cycle phases and a cell-cell transition matrix T (row-stochastic),
    compute a phase-aggregated transition matrix P, quantify forward bias, and (optionally) flip
    the flow orientation so the bias is positive.

    Returns
    -------
    T_oriented : same type as T
        Oriented transition matrix (possibly flipped).
    info : dict
        Diagnostics including P, drift, and whether flipping occurred.

    Notes
    -----
    - T should be N x N with rows corresponding to cells in `adata` (same ordering).
    - For flip_strategy="transpose": returns T.T when drift < 0.
      This is appropriate when your downstream "velocity" uses (T - I)X or sums over j with T_ij.
    - If you already computed a velocity field V and want to flip that instead, use
      flip_strategy="negate_velocity" and flip V := -V yourself (this function will only report).
    """
    import numpy as np

    # --- helper: convert sparse sums to 1D arrays robustly
    def _to_1d(a):
        a = np.asarray(a)
        return a.ravel()

    # --- read phases
    if phase_key not in adata.obs:
        raise KeyError(f"`{phase_key}` not in adata.obs. Available keys include: {list(adata.obs.columns)[:20]} ...")

    phases_raw = adata.obs[phase_key].astype(str).values
    n = adata.n_obs

    # --- basic checks
    if T.shape[0] != n or T.shape[1] != n:
        raise ValueError(f"T must be shape (n_obs, n_obs)=({n},{n}); got {T.shape}")

    # --- map phases -> {0,1,2} in the requested order
    phase_to_idx = {p: i for i, p in enumerate(phase_order)}
    keep = np.array([p in phase_to_idx for p in phases_raw], dtype=bool)

    if verbose:
        dropped = int((~keep).sum())
        if dropped > 0:
            uniq = sorted(set(phases_raw[~keep]))
            print(f"[orient_flow] Dropping {dropped} cells with phases not in {phase_order}: {uniq[:10]}{'...' if len(uniq)>10 else ''}")

    if keep.sum() == 0:
        raise ValueError(f"No cells have phase in {phase_order}. Check `phase_key` and labels.")

    phases = np.array([phase_to_idx[p] for p in phases_raw[keep]], dtype=int)

    # subset T to kept cells (handles dense or scipy sparse)
    Tsub = T[keep][:, keep]
    nsub = Tsub.shape[0]

    # --- compute phase->phase transition matrix P (3x3)
    P = np.zeros((3, 3), dtype=float)

    # Precompute boolean masks for targets
    tgt_masks = [(phases == b) for b in range(3)]

    for a in range(3):
        src_idx = np.where(phases == a)[0]
        if src_idx.size == 0:
            continue

        Ta = Tsub[src_idx]  # rows
        # For each b, compute mean over source rows of mass going to target phase b
        for b in range(3):
            mb = tgt_masks[b]
            if mb.sum() == 0:
                P[a, b] = np.nan
                continue
            # sum over target columns in phase b, per source row, then average
            s = Ta[:, mb].sum(axis=1)
            P[a, b] = float(np.mean(_to_1d(s)))

    # Normalize rows defensively (in case T isn't perfectly row-stochastic after masking)
    row_sums = np.nansum(P, axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        P = P / row_sums

    # --- choose weights
    if use_empirical_weights:
        w = np.array([(phases == a).mean() for a in range(3)], dtype=float)
        w = w / (w.sum() + 1e-12)
    else:
        w = np.ones(3, dtype=float) / 3.0

    # --- signed step matrix delta in {-1,0,+1} for order G1->S->G2M->G1
    delta = np.array([
        [ 0, +1, -1],  # from G1 to (G1,S,G2M)
        [-1,  0, +1],  # from S
        [+1, -1,  0],  # from G2M
    ], dtype=float)

    drift = float(np.nansum(w[:, None] * P * delta))

    flipped = False
    T_oriented = T

    if drift < 0:
        flipped = True
        if flip_strategy == "transpose":
            T_oriented = T.T
        elif flip_strategy == "negate_velocity":
            # We do not modify T in this mode; report the need to negate velocity vectors instead.
            T_oriented = T
        else:
            raise ValueError("flip_strategy must be 'transpose' or 'negate_velocity'")

    info = {
        "phase_order": phase_order,
        "phase_key": phase_key,
        "n_used": int(nsub),
        "weights": w,
        "P_phase": P,
        "drift": drift,
        "flipped": flipped,
        "flip_strategy": flip_strategy,
    }

    if verbose:
        print("[orient_flow] P (rows: from, cols: to) in order", phase_order)
        print(np.array2string(P, precision=3, suppress_small=True))
        print(f"[orient_flow] drift = {drift:.4f}  (positive = forward)")
        if flipped:
            if flip_strategy == "transpose":
                print("[orient_flow] drift<0, returning T.T to orient forward.")
            else:
                print("[orient_flow] drift<0, keep T but you should negate the velocity field V := -V.")
        else:
            print("[orient_flow] drift>=0, keeping T as-is.")

    return T_oriented, info


