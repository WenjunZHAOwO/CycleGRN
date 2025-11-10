import numpy as np

import sys
sys.path.append('../Comparison/')
import numpy as np
import matplotlib.pyplot as plt
from NN_Ours import traj_ours
# from SINDy_func import traj_SINDy
# from NODE_func import traj_NODE
import pickle 
import ot
from scipy.ndimage import gaussian_filter
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import ElasticNet, Ridge, ElasticNetCV, MultiTaskElasticNetCV
from scipy.sparse.linalg import splu

def bin_traj(ys):
    Peq,edges = np.histogramdd(ys , range = [[-4,4],[-4,4]], bins = [20,20],density=True)
    # costM /= costM.max()
    Peq = Peq/sum(Peq.flatten())
    return Peq,edges
    
    
def cost(P1,P2,edges):
    dx = edges[0][1]-edges[0][0]
    dy = edges[1][1] -edges[1][0]
    Xi = [edges[0][0]+dx/2 + dx*i for i in range(len(edges[0])-1)]
    Yi = [edges[1][0]+dy/2 + dx*i for i in range(len(edges[1])-1)]
    xv, yv = np.meshgrid(Xi, Yi, sparse=False, indexing='ij')
    X = np.zeros((len(Xi)*len(Yi),2))
    X[:,0] = xv.reshape(len(Xi)*len(Yi), order='F')
    X[:,1] = yv.reshape(len(Xi)*len(Yi), order='F')
    costM = ot.dist(X, X)
    _,log = ot.lp.emd(P1.flatten(order = 'F'), P2.flatten(order = 'F'), costM, numItermax=1000000, log=True)
    return log['cost']



def train_ours(TIME=500,NAME='RPC-notch.p',seed=0):

    

    TRAJS1 = []
    N_costs = []    
    N_times = []
    ys2,ts2,vs2 = traj_ours(TIME,NAME,seed)

    TRAJS1.append(ys2[:int(1e5)])

    P1,_ = bin_traj(ys2)

    plt.imshow(P1)
    plt.show()
                
    return ys2, ts2, vs2, TRAJS1, P1

import numpy as np
from sklearn.neighbors import NearestNeighbors
import scipy.sparse as sp

def co_moving_kernel_closed_form(
    X, V, k=200, sigma0=None, forward_halfspace=True, eps=1e-12, stimulation=True, threshold=0.95
):
    """
    Closed-form selection of h and sigma^2 using a frozen symmetric kNN kernel P0,
    then construction of the directed co-moving kernel P.

    Inputs
    ------
    X : (n, d) float array        positions
    V : (n, d) float array        velocities
    k : int                       kNN size for graph
    sigma0 : float or None        bandwidth for P0; if None, median kNN distance
    forward_halfspace : bool      keep only neighbors with <v_i, x_j-x_i> > 0
    eps : float                   numerical stabilizer

    Returns
    -------
    h : float                     closed-form global time step
    sigma2 : float                closed-form global variance
    P : (n x n) csr_matrix       directed, row-stochastic co-moving kernel
    P0 : (n x n) csr_matrix      symmetric, row-stochastic frozen kernel
    """
    n, d = X.shape
    k_eff = min(k+1, n)  # include self then drop
    nbr = NearestNeighbors(n_neighbors=k_eff).fit(X)
    dists, idxs = nbr.kneighbors(X)
    # drop self column 0
    idxs = idxs[:, 1:]
    dists = dists[:, 1:]

    # ----- Build frozen symmetric Gaussian kernel K0 and P0 -----
    if sigma0 is None:
        sigma0 = float(np.median(dists) + eps)

    rows, cols, vals = [], [], []
    for i in range(n):
        js = idxs[i]
        Dij = X[js] - X[i]                  # (k, d)
        w = np.exp(-np.sum(Dij*Dij, axis=1) / (2.0 * sigma0 * sigma0))
        rows.extend([i]*len(js)); cols.extend(js); vals.extend(w.tolist())

    K0 = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    rsum0 = np.array(K0.sum(axis=1)).ravel()
    rsum0[rsum0 == 0.0] = 1.0
    P0 = sp.diags(1.0/rsum0) @ K0                      # row-stochastic

    # ----- Closed-form h: h = <Delta, V> / ||V||^2 -----
    Delta = (P0 @ X) - X                               # (n, d)
    num = float(np.sum(Delta * V))                     # sum_i <Delta_i, v_i>
    den = float(np.sum(V * V)) + eps                   # sum_i ||v_i||^2
    h = max(num / den, 0.0)

    # ----- Closed-form sigma^2: expected squared co-moving residual / (d*n) -----
    # Compute sum_j P0_ij * || (x_j - x_i) - h v_i ||^2 efficiently
    sigma2_sum = 0.0
    for i in range(n):
        row_start, row_end = P0.indptr[i], P0.indptr[i+1]
        js = P0.indices[row_start:row_end]
        pij = P0.data[row_start:row_end]               # weights sum to 1
        Rij = (X[js] - X[i]) - h * V[i]               # (deg_i, d)
        sigma2_sum += float(np.sum(pij * np.sum(Rij*Rij, axis=1)))
    sigma2 = max(sigma2_sum / (d * n), eps)

    # ----- Build directed co-moving kernel K using (h, sigma2) -----
    rows, cols, vals = [], [], []
    inv2sig2 = 1.0 / (2.0 * sigma2)
    for i in range(n):
        js = idxs[i]
        Dij = X[js] - X[i]                             # (k, d)
        # optional forward half-space gate
        if forward_halfspace:
            dot = np.sum(V[i] * Dij, axis=1)
            mask = dot > threshold
            if not np.any(mask):
                mask = slice(None)                    # fallback: keep all
        else:
            mask = slice(None)

        R = Dij[mask] - h * V[i]                      # residuals to forecast
        w = np.exp(-np.sum(R*R, axis=1) * inv2sig2)
        kept_js = js[mask]
        rows.extend([i]*len(kept_js)); cols.extend(kept_js); vals.extend(w.tolist())

    K = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    if stimulation:
        K[:,0:100] = 0
    # tiny self-loops to avoid zero rows
    K = K + sp.eye(n, format="csr") * 1e-16
    rsum = np.array(K.sum(axis=1)).ravel()
    rsum[rsum == 0.0] = 1.0
    

    P = sp.diags(1.0/rsum) @ K                         # row-stochastic



    return P

import numpy as np

def pick_k_threshold(
    X, V, #build_laplacian,
    k_list=(100,150,200,250,300,500),
    tau_list=(0.75,0.8,0.9,0.95,0.98,0.99),
    stimulation=True,
    cos_min=0.7,           # per-row cosine floor (direction correctness)
    frac_ok_min=0.9,       # at least this fraction of rows must pass cos_min
    mean_cos_min=0.8,      # overall mean cosine floor
    eps=1e-12
):
    """
    Choose (k, tau) so that:
      (A) directions are correct: most rows have cosine >= cos_min and mean cosine >= mean_cos_min
      (B) among those, displacement E = P@X - X is best explained by a single h*: E ≈ h* V

    Returns: k_best, tau_best, h_best, report
    """
    best = (np.inf, None, None, None)  # (rel_mse, k, tau, h*)
    Vn = np.linalg.norm(V, axis=1, keepdims=True) + eps

    for k in k_list:
        for tau in tau_list:
            P = build_laplacian(X, V, k=k, threshold=tau, stimulation=stimulation)
            E = P @ X - X

            # ----- Direction checks -----
            En = np.linalg.norm(E, axis=1, keepdims=True) + eps
            cos_i = (E * V).sum(axis=1, keepdims=True) / (En * Vn)   # per-row cosine in [-1,1]
            frac_ok = float((cos_i >= cos_min).mean())
            mean_cos = float(cos_i.mean())

            if not (frac_ok >= frac_ok_min and mean_cos >= mean_cos_min):
                continue  # direction not good enough → reject this (k, tau)

            # ----- Single-global-scale fit -----
            num = float((E * V).sum())               # sum_i <E_i, V_i>
            den = float((V * V).sum()) + eps         # sum_i ||V_i||^2
            h_star = max(num / den, 0.0)

            resid = E - h_star * V
            rel_mse = float((resid * resid).sum() / ((E * E).sum() + eps))  # smaller is better
            print(rel_mse)
            print('k='+str(k))
            print('tau='+str(tau))
            if rel_mse < best[0]:
                best = (rel_mse, k, tau, h_star)

    rel_mse_best, k_best, tau_best, h_best = best
    report = dict(rel_mse=rel_mse_best, k=k_best, tau=tau_best, h=h_best,
                  cos_min=cos_min, frac_ok_min=frac_ok_min, mean_cos_min=mean_cos_min)
    return k_best, tau_best#, h_best, report




def build_laplacian(X, V, X_full=None, k=200, threshold=0.95, stimulation=True, return_dense=False, normalize=True):
    # 2. Build a strictly directional transition matrix T
    eps = 1e-10
    if X_full is None:
        X_full = X

    n_cells = X.shape[0]
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X_full)
    _, neighbors = nn.kneighbors(X_full)
    neighbors = neighbors[:, 1:]                 # (n_cells, k)

    # Edge list (rows = sources i, cols = neighbor j)
    rows = np.repeat(np.arange(n_cells), neighbors.shape[1])  # (n_cells*k,)
    cols = neighbors.ravel()                                   # (n_cells*k,)

    # Edge-wise deltas and corresponding V
    delta = X[cols] - X[rows]                    # (n_cells*k, d)
    Vi    = V[rows]                              # (n_cells*k, d)

    # Cosine-like normalized dot product
    num = np.einsum('ij,ij->i', Vi, delta)       # (n_cells*k,)
    nV  = np.linalg.norm(Vi, axis=1) + eps
    nD  = np.linalg.norm(delta, axis=1) + eps
    scores = num / (nV * nD)                     # (n_cells*k,)

    # Threshold edges
    keep = scores > threshold
    data = scores[keep]
    r = rows[keep]
    c = cols[keep]

    # Assemble sparse T and add identity
    T = sp.csr_matrix((data, (r, c)), shape=(n_cells, n_cells))
    T = T + sp.eye(n_cells, format='csr')
    if return_dense:
        T = T.toarray()
    
    if stimulation:
        T[:,0:100] = 0
    
    # Normalize rows
    if normalize:
        T = T / (T.sum(axis=1) + 1e-12)
    
    return T

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu






def resolvent_times_V(T, V, alpha=0.3):
    """
    Compute Y = (I - alpha T)^{-1} V via sparse LU solves (column-wise).
    """
    T = T.tocsc()
    n = T.shape[0]
    A = sp.eye(n, format='csc') - alpha * T
    lu = splu(A)  # factorization reused across RHS
    # Solve for each column of V
    if sp.issparse(V):
        V = V.toarray()
    Y = np.column_stack([lu.solve(V[:, j]) for j in range(V.shape[1])])
    return Y


def find_Lie_derivative(X, T, stimulation=True):
    if stimulation == False:
        return X @ T.T - X
    else:
        V = X @ T.T - X
        
        V[100:,0] = 0
        return V

def time_lagged_correlation(X,V,T, lag=False):
    V = V/(1e-15 + V.std(axis=1).reshape(-1,1) )
    if lag == False:
        G =  V @ T @ V.T
    else:
        print('doing lag')
        G = V @ (resolvent_times_V(T, T @ V.T, alpha=lag))
    G = G - np.diag( np.diag(G))
    G = G/abs(G).max()
    return G

import numpy as np
from statsmodels.stats.multitest import multipletests

# def perm_p_value(X, V, T, n_perm=1000, seed=0, lag=False):
# import numpy as np
# from statsmodels.stats.multitest import multipletests

def perm_p_value(X, V, T, lag=False, n_perm=100, seed=0, fdr_alpha=0.05, dtype=np.float32):
    rng = np.random.default_rng(seed)

    # --- observed ---
    G_obs = time_lagged_correlation(X, V, T, lag=lag).astype(dtype, copy=False)
    abs_obs = np.abs(G_obs)

    
    # --- running count of "worse or equal" null stats ---
    worse = np.zeros_like(G_obs, dtype=np.int32)

    # --- permutations, streamed ---
    for _ in range(n_perm):
        print(_)
        perm = rng.permutation(V.shape[1])  # shuffle cells
        V_perm = V[:, perm]
        G_perm = time_lagged_correlation(X, V_perm, T, lag=lag).astype(dtype, copy=False)

        # update exceedance counts
        worse += (np.abs(G_perm) >= abs_obs)

    # --- empirical p-values ---
    p = (worse + 1) / (n_perm + 1)  # add 1 for smoothing
    p = np.asarray(p, dtype=np.float64)
    # --- FDR correction (BH) ---
    p_flat = p.ravel()
    # print(p_flat)
    # print(p_flat.shape)
    # stop
    # _, p_adj_flat, _, _ = multipletests(p_flat, method='fdr_bh', alpha=fdr_alpha)
    # p_adj = p_adj_flat.reshape(p.shape)
    p_adj = p_flat.reshape(p.shape)

    sig_mask = p_adj < fdr_alpha
    return G_obs, p_adj, sig_mask



import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.covariance import GraphicalLassoCV


def time_lagged_Granger(X,V,T, n_cv=5):
    V = V/(1e-15 + V.std(axis=1).reshape(-1,1) )
    from sklearn.linear_model import ElasticNet, Ridge, ElasticNetCV, MultiTaskElasticNetCV
    V = np.asarray(V, dtype=np.float32)
    if sp.issparse(T):
        T = T.astype(np.float32, copy=False)
    else:
        T = sp.csr_matrix(T.astype(np.float32, copy=False))

    # --- normalize rows of V without keepdims ---
    sc = V.std(axis=1)
    sc[sc < 1e-8] = 1e-8
    V = (V.T / sc).T

    # --- design & targets ---
    X_design = V
    Y_target = (T.T @ V.T).T

    model = MultiTaskElasticNetCV(
        cv=n_cv,
        l1_ratio=[0.01, 0.25, 0.5, 0.75, 1.0],
        n_alphas=40,
        n_jobs=1,
        fit_intercept=False,
        selection='cyclic',
        random_state=0,
    )
    model.fit(X_design, Y_target)

    G = model.coef_.astype(np.float32, copy=False)
    np.fill_diagonal(G, 0.0)
    mx = np.abs(G).max()
    if mx > 0:
        G /= mx
    return G




import os, gc, tempfile
import numpy as np
from scipy import sparse as sp
from sklearn.model_selection import KFold
from sklearn.linear_model import MultiTaskElasticNet, Ridge

def time_lagged_Granger_CV_fast(
    X, V, T,
    n_splits=3,
    l1_grid=[0.01,0.25,0.5,0.75,1.0],
    n_alphas=10,
    alpha_min=1e-4,
    alpha_max=1e0,
    cv_subsample=8000,   # number of cells for CV
    y_block=4096,
    tmp_dir=None,
    dtype=np.float32,
    random_state=0,
    max_iter_cv=500,
    tol_cv=1e-3,
):
    """
    Optimized Granger CV with:
      - Subsampled CV
      - Reduced alpha/l1 grid
      - Looser tolerance for CV fits
      - ❌ No final full refit (returns hyperparams and optional subsample coefficients)
    Works for V shape (d, n_cells), T (n_cells, n_cells).
    """
    # --- Ensure sparse type ---
    if not sp.issparse(T):
        T = sp.csr_matrix(T)
    else:
        T = T.tocsr()
    T = T.astype(dtype, copy=False)

    d, n_cells = V.shape

    # --- Normalize along cell axis ---
    sc = V.std(axis=1)
    sc[sc < 1e-8] = 1e-8
    V = V / sc[:, None]

    
    # 1. Subsample for CV (top entries by |V| sum)
    # ---------------------------
    if cv_subsample < n_cells:
        # Compute activity score for each cell (column)
        activity = np.abs(V).sum(axis=0)  # shape (n_cells,)
        # Get indices of top-k cells
        sub_idx = np.argpartition(activity, -cv_subsample)[-cv_subsample:]
        # (Optional) sort by score for reproducibility
        sub_idx = sub_idx[np.argsort(-activity[sub_idx])]
    else:
        sub_idx = np.arange(n_cells)


    V_sub = V[:, sub_idx]
    T_sub = T[sub_idx, :][:, sub_idx]

    # ---------------------------
    # 2. Build Y_sub = (V_sub @ T_sub).T (memmap)
    # ---------------------------
    if tmp_dir is None:
        tmp_dir = tempfile.gettempdir()
    os.makedirs(tmp_dir, exist_ok=True)
    y_path_sub = os.path.join(tmp_dir, "Y_sub_memmap.dat")
    if os.path.exists(y_path_sub):
        os.remove(y_path_sub)
    Y_sub = np.memmap(y_path_sub, mode='w+', dtype=dtype, shape=(len(sub_idx), d))

    for start in range(0, len(sub_idx), y_block):
        stop = min(start + y_block, len(sub_idx))
        T_block = T_sub[:, start:stop]
        Y_block = (V_sub @ T_block).T
        if Y_block.dtype != dtype:
            Y_block = Y_block.astype(dtype, copy=False)
        Y_sub[start:stop, :] = Y_block
        del Y_block, T_block
        gc.collect()
    Y_sub.flush()

    # ---------------------------
    # 3. CV on subsample
    # ---------------------------
    alphas = np.logspace(np.log10(alpha_max), np.log10(alpha_min), num=n_alphas).astype(dtype)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    best = {'val_mse': np.inf, 'alpha': None, 'l1_ratio': None}

    V_sub_T = V_sub.T  # (n_sub_cells, d)

    for l1 in l1_grid:
        print("starting l1="+str(l1))
        mse_curves = []
        for train_idx, val_idx in kf.split(np.arange(len(sub_idx))):
            X_tr, X_va = V_sub_T[train_idx], V_sub_T[val_idx]
            Y_tr, Y_va = Y_sub[train_idx], Y_sub[val_idx]
            if l1 > 0:
                model = MultiTaskElasticNet(
                    alpha=alphas[0], l1_ratio=l1,
                    fit_intercept=False, warm_start=True,
                    max_iter=max_iter_cv, tol=tol_cv,
                    selection='cyclic', random_state=random_state
                )
            else:
                
                model = Ridge(
                    alpha=alphas[0],
                    fit_intercept=False
                )

            fold_mse = []
            for a in alphas:
                print("starting alpha="+str(a))
                model.set_params(alpha=float(a))
                model.fit(X_tr, Y_tr)
                R = Y_va - X_va @ model.coef_.T
                mse = float(np.mean(R * R))
                fold_mse.append(mse)
                del R
            mse_curves.append(fold_mse)

            del X_tr, X_va, Y_tr, Y_va
            gc.collect()

        mse_curves = np.array(mse_curves)
        mean_mse = mse_curves.mean(axis=0)
        j_best = int(np.argmin(mean_mse))
        if mean_mse[j_best] < best['val_mse']:
            best.update({'val_mse': float(mean_mse[j_best]),
                         'alpha': float(alphas[j_best]),
                         'l1_ratio': float(l1)})

    # ---------------------------
    # 4. Fit on subsample with best params (optional)
    # ---------------------------
    if best['l1_ratio'] > 0:
        model = MultiTaskElasticNet(
            alpha=best['alpha'], l1_ratio=best['l1_ratio'],
            fit_intercept=False, max_iter=1000, tol=1e-4,
            selection='cyclic', random_state=random_state
        )
    else:
        model = Ridge(
            alpha=best['alpha'],
            fit_intercept=False
        )
    model.fit(V_sub_T, Y_sub)
    G_sub = model.coef_.astype(dtype, copy=False)
    np.fill_diagonal(G_sub, 0.0)
    mx = np.abs(G_sub).max()
    if mx > 0:
        G_sub /= mx

    # cleanup subsample Y
    try:
        del Y_sub
        gc.collect()
        os.remove(y_path_sub)
    except Exception:
        pass

    return G_sub, best
