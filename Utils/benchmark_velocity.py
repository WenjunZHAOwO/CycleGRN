import sys
from cycleGRN import bin_traj, train_ours,traj_ours,build_laplacian,find_Lie_derivative
from cycleGRN import time_lagged_correlation, time_lagged_Granger, time_lagged_Granger_CV_fast,pick_k_threshold
import cellrank as cr
from spatial_velocity import orient_flow_by_cell_cycle_phase
import pandas as pd
import numpy as np
import anndata as ad
import scipy.sparse as sp
import scvelo as scv
import scanpy as sc
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns

def compute_scvelo(adata):
    scv.pp.filter_and_normalize(
    adata,
    min_shared_counts=20,
    n_top_genes=2000,
    )
    scv.pp.moments(adata, n_pcs=30, n_neighbors=30)
    scv.tl.recover_dynamics(adata)      # fits gene-wise kinetics; can take a bit
    scv.tl.velocity(adata)#, mode="dynamical")
    scv.tl.velocity_graph(adata)
    sc.tl.umap(adata)
    scv.tl.velocity_pseudotime(adata)
    return



# ---------- Core helpers ----------
def row_normalize(W):
    """Row-normalize adjacency/kernel to a random-walk matrix P."""
    if sp.issparse(W):
        rs = np.asarray(W.sum(axis=1)).ravel() + 1e-12
        return sp.diags(1.0/rs) @ W
    rs = W.sum(axis=1, keepdims=True) + 1e-12
    return W / rs

def flow_from_transition(X2d, P):
    """
    X2d: (n,2) embedding
    P  : (n,n) row-stochastic transition (sparse or dense)
    Returns per-cell flow vectors v = (P @ X - X).
    """
    if sp.issparse(P):
        PX = P @ X2d
    else:
        PX = P.dot(X2d)
    V = PX - X2d
    return V

def interpolate_to_grid(X2d, V, grid_size=60, k=30, bandwidth=None):
    """
    Interpolate per-cell vectors V onto a regular grid over X2d.
    Gaussian-weighted kNN interpolation (bandwidth auto = median NN distance).
    Returns (Xg, Yg, U, V) for streamplot/quiver.
    """
    x, y = X2d[:,0], X2d[:,1]
    xlin = np.linspace(x.min(), x.max(), grid_size)
    ylin = np.linspace(y.min(), y.max(), grid_size)
    Xg, Yg = np.meshgrid(xlin, ylin)
    grid_pts = np.c_[Xg.ravel(), Yg.ravel()]

    nbrs = NearestNeighbors(n_neighbors=min(k, len(X2d)), algorithm='auto').fit(X2d)
    dists, idxs = nbrs.kneighbors(grid_pts, return_distance=True)  # (m,k)
    if bandwidth is None:
        # median of 1-NN distances across grid points
        bandwidth = np.median(dists[:, 0]) + 1e-12

    W = np.exp(- (dists**2) / (2 * bandwidth**2))  # Gaussian weights
    W /= (W.sum(axis=1, keepdims=True) + 1e-12)

    U = (W * V[idxs, 0]).sum(axis=1).reshape(Xg.shape)
    Vv = (W * V[idxs, 1]).sum(axis=1).reshape(Xg.shape)
    return Xg, Yg, U, Vv

# ---------- Visualization ----------
def velocity_streamplot(X2d, V, use_P=True, grid_size=60, k=30, bandwidth=None,
                        density=1.2, minlength=0.5, color_by_speed=True, quiver=False):
    """
    Draw scVelo-like flows on the 2D embedding.
      - If use_P: P_or_W is a row-stochastic transition P.
      - Else:     P_or_W is an adjacency/kernel W (we row-normalize to P).
    """
    
    Xg, Yg, U, Vg = interpolate_to_grid(X2d, V, grid_size=grid_size, k=k, bandwidth=bandwidth)

    speed = np.hypot(U, Vg)
    # plt.figure(figsize=(6,5))
    # background cells
    plt.scatter(X2d[:,0], X2d[:,1], s=6, c="lightgray", alpha=0.7, linewidths=0)

    if quiver:
        step = max(1, grid_size // 25)  # sparser for readability
        Q = plt.quiver(Xg[::step,::step], Yg[::step,::step],
                       U[::step,::step], Vg[::step,::step],
                       speed[::step,::step] if color_by_speed else None,
                       angles='xy', scale_units='xy', scale=1.0, width=0.003)
        if color_by_speed:
            plt.colorbar(Q, label="speed")
    else:
        # streamplot with optional speed colormap
        strm = plt.streamplot(Xg, Yg, U, Vg, density=density, minlength=minlength,
                              color=speed if color_by_speed else 'k', linewidth=1)
        if color_by_speed:
            cbar = plt.colorbar(strm.lines)
            cbar.set_label("speed")


def make_rna_velocity_fig(adata, cycle_field='Cell_cycle_relativePos'):
    vk = cr.kernels.VelocityKernel(adata)
    vk.compute_transition_matrix()
    ck = cr.kernels.ConnectivityKernel(adata)
    ck.compute_transition_matrix()
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    # 1. Run the plot command, but DO NOT save yet. 
    # Set show=False so we can modify it before it renders/closes.
    ax = vk.plot_projection(
        recompute=True, 
        color=cycle_field,
        linewidth=2, 
        title='RNA velocity', 
        fontsize=20, 
        alpha=0.2,
        show=False,   # Important: Keep plot active
        save=None     # Important: Don't save yet
    )
    
    # 2. Get the current figure
    fig = plt.gcf()
    
    # 3. Access the colorbar
    # In these plots, the colorbar is usually the last axis added to the figure.
    cbar_ax = fig.axes[-1] 
    
    # --- OPTION A: Change the Title ---
    cbar_ax.set_ylabel(r'$\theta$', fontsize=12, rotation=0, labelpad=5)
    
    # --- OPTION B: Change Ticks (e.g., to 0, pi, 2pi) ---
    # Assuming your data ranges roughly from 0 to 6.28
    cbar_ax.set_yticks([0, np.pi, 2*np.pi])
    cbar_ax.set_yticklabels(['0', '$\pi$', '$2\pi$'])
    
    return ax, cbar_ax


def make_cyclo_velocity_fig(adata, V, cycle_field='Cell_cycle_relativePos'):
    adata.layers['my_velocity'] = V.T#.tocsr() #np.asarray(V).T

    vk_my = cr.kernels.VelocityKernel(adata, vkey="my_velocity")
    vk_my.compute_transition_matrix()
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    # 1. Run the plot command, but DO NOT save yet. 
    # Set show=False so we can modify it before it renders/closes.
    ax = vk_my.plot_projection(
        recompute=True, 
        color=cycle_field,
        linewidth=3, 
        title='CYCLO velocity', 
        fontsize=20, 
        alpha=0.2,
        show=False,   # Important: Keep plot active
        save=None     # Important: Don't save yet
    )
    
    # 2. Get the current figure
    fig = plt.gcf()
    
    # 3. Access the colorbar
    # In these plots, the colorbar is usually the last axis added to the figure.
    cbar_ax = fig.axes[-1] 
    
    # --- OPTION A: Change the Title ---
    cbar_ax.set_ylabel(r'$\theta$', fontsize=12, rotation=0, labelpad=5)
    
    # --- OPTION B: Change Ticks (e.g., to 0, pi, 2pi) ---
    # Assuming your data ranges roughly from 0 to 6.28
    cbar_ax.set_yticks([0, np.pi, 2*np.pi])
    cbar_ax.set_yticklabels(['0', '$\pi$', '$2\pi$'])
    
    return ax, cbar_ax

import numpy as np
from scipy.sparse import issparse
import ot  # POT

def _get_dense(adata, idx, use="X", layer=None, obsm_key=None, genes=None):
    """
    Fetch data matrix for selected rows:
      - gene matrix: use="genes" via layer or X, optionally subset genes
      - cost matrix: obsm_key (e.g. X_pca)
    Returns a dense numpy array.
    """
    if obsm_key is not None:
        return np.asarray(adata.obsm[obsm_key][idx])

    if layer is not None:
        M = adata.layers[layer][idx]
    else:
        M = adata.X[idx]

    if issparse(M):
        M = M.toarray()
    else:
        M = np.asarray(M)

    if genes is not None:
        M = M[:, genes]
    return M


def _ot_plan_barycentric(Xs_cost, Xt_cost, Xt_gene, reg=0.05, scale_cost=True):
    """
    Compute entropic OT plan between source/target in cost-space,
    then barycentric projection of target GENE space onto each source cell.

    Returns:
      Yhat_gene: (n_source, n_genes)
    """
    ns = Xs_cost.shape[0]
    nt = Xt_cost.shape[0]
    a = np.ones(ns) / ns
    b = np.ones(nt) / nt

    C = ot.dist(Xs_cost, Xt_cost, metric="euclidean") ** 2
    if scale_cost:
        C = C / (np.median(C) + 1e-12)

    pi = ot.sinkhorn(a, b, C, reg=reg, numItermax=2000, stopThr=1e-9)

    # barycentric projection in gene space: yhat_i = (pi_i · Xt_gene) / a_i
    Yhat_gene = (pi @ Xt_gene) / a[:, None]
    return Yhat_gene


def ot_velocity_gene_centered_cyclic(
    adata,
    phase_key="Cell_cycle_relativePos",
    n_bins=20,
    cost_rep="X_pca",      # geometry for OT cost (obsm key)
    gene_layer=None,       # gene expression source: None -> adata.X, else adata.layers[gene_layer]
    genes=None,            # None -> all genes; or pass indices (int array) or boolean mask
    reg=0.05,
    max_bin_cells=400,
    min_bin_cells=30,
    rng_seed=0,
    store_layer="ot_velocity",
    dtype=np.float32,
    return_debug=False,
):
    """
    Compute gene-level OT velocity V (n_cells, n_genes_selected) on a circular pseudotime
    grid with (generally) non-uniform bin spacing (quantile bins).

    We use the *non-uniform* centered derivative at bin k, written as a weighted average
    of forward/backward slopes:

        h_f = t_{k+1} - t_k  (wrapped to (0,1])
        h_b = t_k - t_{k-1}  (wrapped to (0,1])

        v ≈ (h_b/(h_f+h_b)) * (f_{k+1}-f_k)/h_f  +  (h_f/(h_f+h_b)) * (f_k-f_{k-1})/h_b

    Here:
      f_k      = Xi_gene (current bin gene expression)
      f_{k+1}  = yhat_f  (OT barycentric projection onto next bin)
      f_{k-1}  = yhat_b  (OT barycentric projection onto prev bin)

    OT plans are computed in COST space (cost_rep) and barycentric projections are in GENE space.
    """
    rng = np.random.default_rng(rng_seed)

    phase = adata.obs[phase_key].to_numpy()
    phase = np.mod(phase, 1.0)
    n = adata.n_obs

    # interpret genes selector
    if genes is None:
        gene_idx = None
        n_genes_out = adata.n_vars
    else:
        if isinstance(genes, (np.ndarray, list, tuple)) and np.asarray(genes).dtype == bool:
            gene_idx = np.where(np.asarray(genes))[0]
        else:
            gene_idx = np.asarray(genes, dtype=int)
        n_genes_out = len(gene_idx)

    # quantile binning (balanced bin sizes)
    edges = np.quantile(phase, np.linspace(0, 1, n_bins + 1))
    edges[0] = 0.0
    edges[-1] = 1.0
    bin_id = np.digitize(phase, edges, right=False) - 1
    bin_id = np.clip(bin_id, 0, n_bins - 1)

    # bin centers in circular coordinate
    centers = np.mod(0.5 * (edges[:-1] + edges[1:]), 1.0)

    def forward_wrap(delta):
        """Map delta to (0,1] on the circle."""
        d = delta % 1.0
        return d if d > 0 else 1.0

    # indices per bin (with optional subsampling)
    bins = []
    for b in range(n_bins):
        idx = np.where(bin_id == b)[0]
        if idx.size == 0:
            bins.append(idx)
            continue
        if max_bin_cells is not None and idx.size > max_bin_cells:
            idx = rng.choice(idx, size=max_bin_cells, replace=False)
        bins.append(np.sort(idx))

    # output (dense). If this is too big, restrict genes (highly recommended).
    V = np.full((n, n_genes_out), np.nan, dtype=dtype)

    debug = {"edges": edges, "centers": centers, "bin_sizes": [len(x) for x in bins], "used": []}

    for b in range(n_bins):
        I = bins[b]
        if I.size < min_bin_cells:
            continue

        b_f = (b + 1) % n_bins
        b_b = (b - 1) % n_bins
        Jf = bins[b_f]
        Jb = bins[b_b]

        # non-uniform phase steps (positive, wrapped)
        h_f = forward_wrap(centers[b_f] - centers[b])   # t_{k+1} - t_k
        h_b = forward_wrap(centers[b]   - centers[b_b]) # t_k - t_{k-1}

        # source geometry + genes
        Xi_cost = _get_dense(adata, I, obsm_key=cost_rep)
        Xi_gene = _get_dense(adata, I, layer=gene_layer, genes=gene_idx)

        yhat_f = None
        yhat_b = None

        if Jf.size >= min_bin_cells:
            Xf_cost = _get_dense(adata, Jf, obsm_key=cost_rep)
            Xf_gene = _get_dense(adata, Jf, layer=gene_layer, genes=gene_idx)
            yhat_f = _ot_plan_barycentric(Xi_cost, Xf_cost, Xf_gene, reg=reg)

        if Jb.size >= min_bin_cells:
            Xb_cost = _get_dense(adata, Jb, obsm_key=cost_rep)
            Xb_gene = _get_dense(adata, Jb, layer=gene_layer, genes=gene_idx)
            yhat_b = _ot_plan_barycentric(Xi_cost, Xb_cost, Xb_gene, reg=reg)

        if (yhat_f is not None) and (yhat_b is not None):
            # weighted average of forward/backward slopes on non-uniform grid
            slope_f = (yhat_f - Xi_gene) / h_f
            slope_b = (Xi_gene - yhat_b) / h_b
            w_f = h_b / (h_f + h_b)
            w_b = h_f / (h_f + h_b)
            v = w_f * slope_f + w_b * slope_b
            V[I] = v.astype(dtype, copy=False)
            if return_debug:
                debug["used"].append((b, "centered_nonuniform", int(I.size), int(Jb.size), int(Jf.size)))

        elif yhat_f is not None:
            v = (yhat_f - Xi_gene) / h_f
            V[I] = v.astype(dtype, copy=False)
            if return_debug:
                debug["used"].append((b, "forward", int(I.size), int(Jf.size)))

        elif yhat_b is not None:
            v = (Xi_gene - yhat_b) / h_b
            V[I] = v.astype(dtype, copy=False)
            if return_debug:
                debug["used"].append((b, "backward", int(I.size), int(Jb.size)))

        else:
            continue

    # store as layer; shape is (n_cells, n_genes_selected)
    adata.layers[store_layer] = V

    if return_debug:
        return V, debug
    return V

def make_ot_velocity_fig(adata, V_OT, cycle_field='Cell_cycle_relativePos'):

    

    adata.layers['OT_velocity'] = V_OT#.tocsr() #np.asarray(V).T
    
    vk_my = cr.kernels.VelocityKernel(adata, vkey="OT_velocity")
    vk_my.compute_transition_matrix()
    
    # 1. Run the plot command, but DO NOT save yet. 
    # Set show=False so we can modify it before it renders/closes.
    ax = vk_my.plot_projection(
        recompute=True, 
        color=cycle_field,
        linewidth=3, 
        title='OT velocity', 
        fontsize=20, 
        alpha=0.2,
        show=False,   # Important: Keep plot active
        save=None     # Important: Don't save yet
    )
    
    # 2. Get the current figure
    fig = plt.gcf()
    
    # 3. Access the colorbar
    # In these plots, the colorbar is usually the last axis added to the figure.
    cbar_ax = fig.axes[-1] 
    
    # --- OPTION A: Change the Title ---
    cbar_ax.set_ylabel(r'$\theta$', fontsize=12, rotation=0, labelpad=5)
    
    # --- OPTION B: Change Ticks (e.g., to 0, pi, 2pi) ---
    # Assuming your data ranges roughly from 0 to 6.28
    cbar_ax.set_yticks([0, np.pi, 2*np.pi])
    cbar_ax.set_yticklabels(['0', '$\pi$', '$2\pi$'])
    return ax, cbar_ax


def compare_gene_all(adata, gene_id, V,  figsize=(12,3)):
    
    fig = plt.figure(figsize=figsize)  # width, height in inches
    
    # --- 1. Gene Expression ---
    ax1 = plt.subplot(1, 3, 1)
    c = np.array(adata[:, adata.var_names[gene_id]].X.todense()).reshape(-1)
    order = np.argsort(abs(c))
    sc1 = plt.scatter(
        adata.obsm['X_umap'][order, 0],
        adata.obsm['X_umap'][order, 1],
        c=c[order],
        s=3
    )
    cbar = plt.colorbar(
        sc1, 
        ax=ax1,            # <--- Attach to the 3rd plot
        fraction=0.046, 
        pad=0.04, 
        shrink=0.4,        # <--- 40% height (Tiny)
        aspect=30          # <--- Thinner bar
    )
    
    plt.title("Gene expression \n" + adata.var_names[gene_id])
    plt.axis('off')
    
    # --- 2. scVelo Velocity ---
    ax2 = plt.subplot(1, 3, 2)
    order = np.argsort(adata.layers['velocity'][:, gene_id])
    sc3 = plt.scatter(
        adata.obsm['X_umap'][order, 0],
        adata.obsm['X_umap'][order, 1],
        c=adata.layers['velocity'][order, gene_id],
        cmap='seismic',
        s=3
    )
    plt.clim(
        -5 * np.std(adata.layers['velocity'][order, gene_id]),
         5 * np.std(adata.layers['velocity'][order, gene_id])
    )
    plt.title("RNA velocity \n" + adata.var_names[gene_id])
    plt.axis('off')
    
    # --- THE TINY COLORBAR ---
    # We attach it explicitly to 'ax3' (the current subplot)
    cbar = plt.colorbar(
        sc3, 
        ax=ax2,            # <--- Attach to the 3rd plot
        fraction=0.046, 
        pad=0.04, 
        shrink=0.4,        # <--- 40% height (Tiny)
        aspect=30          # <--- Thinner bar
    )
    
    # --- 3. Our Velocity ---
    ax3 = plt.subplot(1, 3, 3)
    order = np.argsort(abs(V[gene_id, :]))
    sc4 = plt.scatter(
        adata.obsm['X_umap'][order, 0],
        adata.obsm['X_umap'][order, 1],
        c=V[gene_id, order],
        cmap='seismic',
        s=3
    )
    plt.clim(
        -5 * np.std(V[gene_id, order]),
         5 * np.std(V[gene_id, order])
    )
    plt.title("CYCLO velocity \n" + adata.var_names[gene_id])
    plt.axis('off')
    
    # --- THE TINY COLORBAR ---
    # We attach it explicitly to 'ax3' (the current subplot)
    cbar = plt.colorbar(
        sc4, 
        ax=ax3,            # <--- Attach to the 3rd plot
        fraction=0.046, 
        pad=0.04, 
        shrink=0.4,        # <--- 40% height (Tiny)
        aspect=30          # <--- Thinner bar
    )

from scipy.ndimage import gaussian_filter1d

def circular_smooth(x, y, sigma=40):
    x = np.asarray(x); y = np.asarray(y)
    order = np.argsort(x)
    x0 = x[order]; y0 = y[order]

    # duplicate data: [x, x+1, x+2]
    x3 = np.concatenate([x0, x0 + 1, x0 + 2])
    y3 = np.concatenate([y0, y0, y0])

    y3s = gaussian_filter1d(y3, sigma=sigma, mode="nearest")

    # take the middle copy (x in [1,2]) and shift back to [0,1]
    mid = (x3 >= 1) & (x3 <= 2)
    x_mid = x3[mid] - 1
    y_mid = y3s[mid]

    # restore increasing x in [0,1]
    ord2 = np.argsort(x_mid)
    return x_mid[ord2], y_mid[ord2]
    
def compare_gene_mean(adata, gene_id, V, figsize=(14,3.5), cycle_field='Cell_cycle_relativePos', symbol=None):
    def symmetric_ylim(y, q=0.99, pad=1.1):
        m = np.quantile(np.abs(y), q)
        return (-pad * m, pad * m)
    
    
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    
    # Setup for publication aesthetics
    plt.rcParams['font.family'] = 'sans-serif' # Arial/Helvetica are standard
    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.linewidth'] = 1.2
    
    # Define data inputs (Mapping title to data source)
    # calculating these upfront makes the plotting loop cleaner
    x = adata.obs[cycle_field].to_numpy()/2/np.pi
    order = np.argsort(x)
    y_expr = np.array(adata[:, adata.var_names[gene_id]].X.todense()).reshape(-1)
    y_scvelo = np.asarray(adata.layers['velocity'][:, gene_id]).ravel()
    y_our = V[gene_id, :]
    # y_ot = V_OT[:,gene_id]
    
    data_map = [
        {"data": y_expr, "title": "Expression", "ylabel": "Norm. Expression", "is_vel": False, "color": "black"},
        {"data": y_scvelo, "title": "RNA velocity", "ylabel": "Velocity", "is_vel": True, "color": "grey"},
        {"data": y_our, "title": "CYCLO velocity", "ylabel": "Velocity", "is_vel": True, "color": "#d62728"}, # Highlight your method with color
       # {"data": y_ot, "title": "OT velocity", "ylabel": "Velocity", "is_vel": True, "color": 'tab:blue'} # Highlight your method with color
    
    ]
    
    # Create Figure
    fig, axes = plt.subplots(1, len(data_map), figsize=figsize, constrained_layout=True)
    
    # Main Title (The Gene Name)
    if symbol is not None:
        gene_name = adata.var['gene_symbol'][gene_id]
    else:
        gene_name = adata.var_names[gene_id]
    fig.suptitle(f"{gene_name} Dynamics", fontsize=18, weight='bold', y=1.1)
    
    for ax, d in zip(axes, data_map):
        y = d["data"] 
        
        # 1. Plot Scatter (Rasterized for PDF export efficiency)
        ax.scatter(x* 2 * np.pi, y, s=4, alpha=0.25, color="slategray", rasterized=True, edgecolor='none')
        
        # 2. Plot Trend Line
        xs, ys = circular_smooth(x[order], y[order], sigma=50)
        ax.plot(xs* 2 * np.pi, ys, lw=2.5, color=d["color"], label='Trend')
        
        # 3. Add Zero Line for velocities
        if d["is_vel"]:
            ax.axhline(0, lw=1, linestyle='--', color="black", alpha=0.5, zorder=0)
            ax.set_ylim(*symmetric_ylim(y))
            
        # 4. Aesthetics
        ax.set_title(d["title"], fontsize=16, pad=10)
        ax.set_xlabel("Cell cycle position (θ)")
        
        # Only show Y label if helpful, or simplistic styling
        ax.set_ylabel(d["ylabel"], fontsize=12)
        
        # Remove top and right spines (classic scientific style)
        sns.despine(ax=ax)
    
    phases = [
        (0.0, 0.33*2*np.pi, "G1", "#e0f2f1"),   # Light Teal
        (0.33*2*np.pi, 0.66*2*np.pi, "S", "#fff3e0"),   # Light Orange
        (0.66*2*np.pi, 2*np.pi, "G2/M", "#fce4ec")  # Light Pink
    ]
    
    for ax in axes:
        ax.set_xlim([0,2*np.pi])
        for start, end, label, color in phases:
            # Add colored background
            ax.axvspan(start, end, color=color, alpha=0.3, zorder=-10, lw=0)
            
            # Optional: Add text labels at the top of the plot
            # only do this on the middle plot to avoid clutter
            if False:#x == axes[0]: 
                mid_point = (start + end) / 2
                ax.text(mid_point, ax.get_ylim()[1] - 0.2, label, 
                        ha='center', va='bottom', fontsize=10, color='black')


def compute_sign_agreement(adata, V, V_OT):
    import numpy as np

    V_my = np.asarray(V)                      # (n_genes, n_cells) expected
    V_sc = adata.layers["velocity"]           # (n_cells, n_genes)
    
    # ensure dense 2D ndarray for V_sc (only if needed)
    V_sc = np.asarray(V_sc)                   # may be big; if too big, do gene-by-gene
    
    q = 0.9
    min_support = 30  # require at least this many cells passing the mask per gene
    
    agree = np.full(V_my.shape[0], np.nan)
    support = np.zeros(V_my.shape[0], dtype=int)
    
    for g in range(V_my.shape[0]):
        v_my = V_my[g, :]            # (n_cells,)
        v_sc = V_sc[:, g]            # (n_cells,)
    
        thr_sc = np.quantile(np.abs(v_sc), q)
        thr_my = np.quantile(np.abs(v_my), q)
    
        mask = (np.abs(v_sc) > thr_sc) & (np.abs(v_my) > thr_my)
        m = int(mask.sum())
        support[g] = m
        if m < min_support:
            continue
    
        agree[g] = np.mean((v_my[mask] * v_sc[mask]) >= 0)
    
    rng = np.random.default_rng(0)
    agree_null = np.full(V_my.shape[0], np.nan)
    
    for g in range(V_my.shape[0]):
        v_my = V_my[g, :]
        v_sc = V_sc[:, g]
        thr_sc = np.quantile(np.abs(v_sc), q)
        thr_my = np.quantile(np.abs(v_my), q)
        mask = (np.abs(v_sc) > thr_sc) & (np.abs(v_my) > thr_my)
        m = int(mask.sum())
        if m < min_support:
            continue
        v_sc_shuf = v_sc[rng.permutation(v_sc.shape[0])]
        agree_null[g] = np.mean((v_my[mask] * v_sc_shuf[mask]) >= 0)
    
    
    import numpy as np
    
    V_ot = np.asarray(V_OT.T)                      # (n_genes, n_cells) expected
    
    # ensure dense 2D ndarray for V_sc (only if needed)
    V_sc = np.asarray(V_sc)                   # may be big; if too big, do gene-by-gene
    
    q = 0.5
    min_support = 30  # require at least this many cells passing the mask per gene
    
    agree_ot = np.full(V_ot.shape[0], np.nan)
    support_ot = np.zeros(V_ot.shape[0], dtype=int)
    
    for g in range(V_my.shape[0]):
        v_ot = V_ot[g, :]            # (n_cells,)
        v_sc = V_sc[:, g]            # (n_cells,)
    
        thr_sc = np.quantile(np.abs(v_sc), q)
        thr_ot = np.quantile(np.abs(v_ot), q)
    
        mask = (np.abs(v_sc) > thr_sc) & (np.abs(v_ot) > thr_ot)
        m = int(mask.sum())
        support_ot[g] = m
        if m < min_support:
            continue
    
        agree_ot[g] = np.mean((v_ot[mask] * v_sc[mask]) >= 0)
    return agree, agree_ot, agree_null

def make_agreement_hist(adata, agree, agree_ot, agree_null):
    

    # remove NaNs
    agree_clean = agree[~np.isnan(agree)]
    agree_null_clean = agree_null[~np.isnan(agree_null)]
    
    bins = np.linspace(0, 1, 31)
    plt.rcParams['font.size'] = 10
    plt.figure(figsize=(4, 3))
    
    med_obs = np.nanmedian(agree_clean)
    med_null = np.nanmedian(agree_null_clean)
    med_ot = np.nanmedian(agree_ot)
    
    # plt.hist(
    #     agree_null_clean,
    #     bins=bins,
    #     density=True,
    #     alpha=0.3,
    #     label=f"Baseline",
    #     color='slategray',
    # )
    
    plt.hist(
        agree_clean,
        bins=bins,
        density=True,
        alpha=0.3,
        color='red',
        label=f"CYCLO velocity",
    )
    
    plt.hist(
        agree_ot,
        bins=bins,
        density=True,
        alpha=0.3,
        color='tab:blue',
        label=f"OT Velocity",
    )
    
    # vertical median lines
    # plt.axvline(med_null, color="slategray", linestyle="--", lw=2)
    plt.axvline(med_obs, color="red", linestyle="--", lw=2)
    plt.axvline(med_ot, color="tab:blue", linestyle="--", lw=2)
    # annotate medians
    ymax = plt.ylim()[1]
    
    # plt.text(
    #     med_null + 0.01,
    #     0.9 * ymax,
    #     f"Median = {med_null:.2f}",
    #     color="slategray",
    #     ha="left",
    #     va="top",
    # )
    
    plt.text(
        med_obs + 0.01,
        0.75 * ymax,
        f"Median = {med_obs:.2f}",
        color="red",
        ha="left",
        va="top",
    )
    
    plt.text(
        med_ot - 0.2,
        0.5 * ymax,
        f"Median = {med_ot:.2f}",
        color="tab:blue",
        ha="left",
        va="top",
    )
    
    plt.legend(frameon=False)
    plt.title("Sign agreement among \n high-magnitude velocity entries")


def rank_genes_by_coherence_clean(adata, velocity_layer='velocity', theta_key='Cell_cycle_relativePos', min_cells=100):
    scores = []
    
    # 1. Identify genes to SKIP
    ignore_prefixes = ('MT-', 'RPS', 'RPL', 'MRPS', 'MRPL', 'None') 
    candidates = [g for g in adata.var_names if not g.startswith(ignore_prefixes)]
    
    # --- AUTO-DETECT PHASE SCALE ---
    # Check if the data is already in radians (0 to 6.28) or normalized (0 to 1)
    all_theta = adata.obs[theta_key].values
    max_theta = np.max(all_theta)
    
    # If max value is significantly > 1, assume it's already radians
    is_radians = max_theta > 1.1
    
    print(f"Ranking {len(candidates)} genes...")
    print(f"Detected phase max: {max_theta:.4f}. Treating as {'Radians (0-2pi)' if is_radians else 'Normalized (0-1)'}.")
    # -------------------------------

    for gene in candidates:
        # Get Velocity
        v = adata[:, gene].layers[velocity_layer]
        if hasattr(v, "todense"): v = v.todense().flatten()
        
        # --- SPARSITY FILTER ---
        # Skip genes that are non-zero in fewer than 'min_cells'
        # This removes "ghost" genes like TEX36 that only exist in 1-2 cells
        if np.count_nonzero(v) < min_cells:
            continue

        # Get Phase
        theta = adata.obs[theta_key].values
        
        # --- PHASE SCALING CHECK ---
        # If data is 0-1, we multiply by 2pi. If it's already radians, we leave it.
        if is_radians:
            complex_phase = np.exp(1j * theta)
        else:
            complex_phase = np.exp(1j * theta * 2 * np.pi) 
        
        # --- METRIC: Weighted Sum ---
        weighted_sum = np.sum(v * complex_phase)
        
        # Normalize
        score = np.abs(weighted_sum) / np.sum(np.abs(v))
            
        scores.append({'gene': gene, 'coherence_score': score, 'n_cells': np.count_nonzero(v)})
        
    return pd.DataFrame(scores).sort_values('coherence_score', ascending=False)


def nominate_driver_genes(adata, gene_ranks, gene_ranks_my, gene_ranks_gene, cycle_field='Cell_cycle_relativePos'):
    
    
    # 1. Define the two lists
    scvelo_genes = gene_ranks['gene'].head(10).tolist()
    our_genes = gene_ranks_my['gene'].head(10).tolist()
    count_genes = gene_ranks_gene['gene'].head(10).tolist()
    
    # Helper function that returns the matrix AND the peak location
    def get_heatmap_data(genes, adata):
        n_bins = 100
        bins = np.linspace(0, 1, n_bins)
        matrix = []
        names = []
        peak_positions = []
        
        for gene in genes:
            try:
                # Get Expression Data
                data = np.array(adata[:, gene].X.todense()).flatten()
            except:
                continue
                
            theta = adata.obs[cycle_field].values
            
            # Binning & Smoothing
            digitized = np.digitize(theta, bins)
            smooth_curve = []
            for i in range(1, len(bins)):
                mask = digitized == i
                if np.any(mask):
                    smooth_curve.append(np.mean(data[mask]))
                else:
                    smooth_curve.append(0)
            
            # Smooth the line to find a clear peak
            smooth_series = pd.Series(smooth_curve).rolling(window=10, center=True, min_periods=1).mean().values
            
            # Normalize (Z-score)
            if np.std(smooth_series) == 0:
                z_score = smooth_series
            else:
                z_score = (smooth_series - np.mean(smooth_series)) / np.std(smooth_series)
                
            matrix.append(z_score)
            names.append(gene)
            # Find index of max value (Peak Phase)
            peak_positions.append(np.argmax(smooth_series))
        
        return np.array(matrix), names, np.array(peak_positions)
    
    # 2. Get Data
    mat_scvelo, names_scvelo, peaks_scvelo = get_heatmap_data(scvelo_genes, adata)
    mat_ours, names_ours, peaks_ours = get_heatmap_data(our_genes, adata)
    mat_count, names_count, peaks_count = get_heatmap_data(count_genes, adata)
    
    # 3. SORT BY PEAK (The Critical Step)
    # We zip them together, sort by peak index, and unzip
    order_scvelo = np.argsort(peaks_scvelo)
    mat_scvelo_sorted = mat_scvelo[order_scvelo]
    names_scvelo_sorted = [names_scvelo[i] for i in order_scvelo]
    
    order_ours = np.argsort(peaks_ours)
    mat_ours_sorted = mat_ours[order_ours]
    names_ours_sorted = [names_ours[i] for i in order_ours]
    
    order_count = np.argsort(peaks_count)
    mat_count_sorted = mat_count[order_count]
    names_count_sorted = [names_count[i] for i in order_count]
    
    # 4. Plot
    # Increase width slightly (16) to accommodate the side bar
    fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharey=False) 
    limit = 5.5
    font_size = 14
    
    sns.heatmap(
        mat_count_sorted, 
        cmap='seismic', 
        ax=axes[0], 
        cbar=False,            # Turn off internal colorbar
        xticklabels=False, 
        yticklabels=names_count_sorted,
        center=0, vmin=-limit, vmax=limit
    )
    axes[0].set_title("Counts: Top Ranked Genes\n(Sorted by Peak Time)", fontsize=16)
    axes[0].set_xlabel("Cell Cycle Position", fontsize=14)
    axes[0].tick_params(axis='y', labelsize=font_size)
    
    # --- PLOT 2: Our Method (Link Colorbar to external axis) ---
    # Create a new axis for the colorbar [left, bottom, width, height] in figure coordinates
    # 0.92 = 92% across the page (far right)
    # cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7]) 
    
    # --- PLOT 1: scVelo (No Colorbar) ---
    sns.heatmap(
        mat_scvelo_sorted, 
        cmap='seismic', 
        ax=axes[1], 
        cbar=False,            # Turn off internal colorbar
        xticklabels=False, 
        yticklabels=names_scvelo_sorted,
        center=0, vmin=-limit, vmax=limit
    )
    axes[1].set_title("RNA Velocity: Top Ranked Genes\n(Sorted by Peak Time)", fontsize=16)
    axes[1].set_xlabel("Cell Cycle Position", fontsize=14)
    axes[1].tick_params(axis='y', labelsize=font_size)
    
    # --- PLOT 2: Our Method (Link Colorbar to external axis) ---
    # Create a new axis for the colorbar [left, bottom, width, height] in figure coordinates
    # 0.92 = 92% across the page (far right)
    # cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7]) 
    
    sns.heatmap(
        mat_ours_sorted, 
        cmap='seismic', 
        ax=axes[2], 
        cbar=False,             # Turn on...
        # cbar_ax=cbar_ax,       # ...but draw it in this specific box
        xticklabels=False, 
        yticklabels=names_ours_sorted,
        center=0, vmin=-limit, vmax=limit, 
        # cbar_kws={'label': 'Z-scored Expression'}
    )
    axes[2].set_title("CYCLO: Top Ranked Genes\n(Sorted by Peak Time)", fontsize=16)
    axes[2].set_xlabel("Cell Cycle Position", fontsize=14)
    axes[2].tick_params(axis='y', labelsize=font_size)
    
    # Style the separate colorbar
    # cbar_ax.tick_params(labelsize=12)
    # cbar_ax.yaxis.label.set_size(14)
    
    # Adjust layout to leave room on the right for the bar
    # plt.subplots_adjust(right=0.8) 
    plt.tight_layout()
    
    
