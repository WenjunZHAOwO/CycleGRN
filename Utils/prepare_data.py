# %matplotlib notebook
import numpy as np
import pandas as pd
import sys
sys.path.append('/Users/wenjunzhao/dropbox/OT-Velocity-local/Utils/')
sys.path.append('/Users/wenjunzhao/dropbox/cardamom/results_article/Benchmark_on_simulated_data/_scripts/')
from sincerities import *

from utils_Velo import *
#from agw_scootr import *
from utils import *
import scipy
import scanpy
import matplotlib.pyplot as plt
# from utils_BEELINE_scRNAseq import *

def prepare_data( cell_stage, gene_subset = False, q_num=10, data_dir='../../mouseprogenitormtxs/' ):
    if cell_stage == 'early' or cell_stage == 'late':
        df = scanpy.read_mtx(data_dir+cell_stage+'RPCs_mretdev.mtx')
        df = df.T
        pheno = pd.read_csv(data_dir+cell_stage+'RPCs_pheno.csv', delimiter=',',index_col=0)
        df.obs_names = pheno.index
    else:
        df = scanpy.read_mtx(data_dir+cell_stage+'_mretdev.mtx')
        df = df.T
        pheno = pd.read_csv(data_dir+cell_stage+'_pheno.csv', delimiter=',',index_col=0)
        df.obs_names = pheno.index
    genes = pd.read_csv(data_dir+'10x_mouse_retina_development_feature.csv',index_col = 0)
    df.var_names = genes.index
    df.obs['age'] = pheno['age']
    
    df.obs['umap_coord1'] = pheno['umap_coord1']
    df.obs['umap_coord2'] = pheno['umap_coord2']
    df.obs['umap_coord3'] = pheno['umap_coord3']
    
    ages = ['E11', 'E12', 'E14', 'E16', 'E18', 'P0', 'P14', 'P2', 'P5', 'P8']
    ages_values = [11,12,14,16,18,20,34,22,25,28]
    df.obs['age (days)'] = 0
    for i,age in enumerate(ages):
        
        idx = df[df.obs['age'] == age].obs.index.values.tolist()
        
        
        # for cell in idx:
        #     df.obs['age (days)'][cell] = ages_values[i]
        df.obs.loc[idx, 'age (days)'] = ages_values[i]
    df.obs['bin'] = df.obs['age (days)']
    
    scanpy.pp.log1p(df)
    # scanpy.pp.neighbors(df)
    # scanpy.tl.umap(df)
    cell_names_to_subset = np.where( df.obs['age (days)'] < 21 )[0]
    len(cell_names_to_subset)
    #data_early_subset = df[cell_names_to_subset,gene_names_to_subset ]
    data_subset = df[cell_names_to_subset,: ]

    if cell_stage == 'early' or cell_stage == 'late':
        tricycle_time = pd.read_csv(data_dir+'/clark_214689_continuous_theta.csv', delimiter=',',index_col=0)
        
        data_subset.obs['Tricycle'] = tricycle_time['tricyclePosition']
        data_subset.obs['tricycle-qcut'] = pd.qcut(data_subset.obs['Tricycle'], q=q_num)
        
        theta_time = pd.read_csv(data_dir+'/clark_214689_continuous_theta.csv', delimiter=',',index_col=0)
        data_subset.obs['theta'] = theta_time['theta_pi']
        data_subset.obs['theta-qcut'] = pd.qcut(data_subset.obs['theta'], q=q_num)

    if gene_subset == True:
        genes_subset = pd.read_csv(data_dir+'/genes_for_subset.csv',index_col = 0)
        gene_names_to_subset = [str(idx) for idx in genes_subset['x'] ]
        if cell_stage == 'early' or cell_stage == 'late':
            data_subset = data_subset[data_subset.obs['Tricycle'].notna()]
        return data_subset[:,gene_names_to_subset]
    else:
        return data_subset


def run_sincerities( df, time = 'days', save=False, saveName=None ):

    pandas_data = df.X.todense()
    if time == 'days':
        pandas_data = np.concatenate( ( np.array(df.obs['age (days)']).reshape(-1,1), pandas_data), axis=1 )
    elif time == 'tricycle':
        df.obs['bin'] = df.obs['Tricycle']
        for i in range(df.obs.shape[0]):
            if ~np.isnan(df.obs['Tricycle'][i]):
                df.obs['bin'][i] = df.obs['tricycle-qcut'][i].mid
        pandas_data = np.concatenate( ( np.array(df.obs['bin']).reshape(-1,1), pandas_data), axis=1 )
    else:
        df.obs['bin'] = df.obs['theta']
        for i in range(df.obs.shape[0]):
            if ~np.isnan(df.obs['Tricycle'][i]):
                df.obs['bin'][i] = df.obs['theta-qcut'][i].mid
        pandas_data = np.concatenate( ( np.array(df.obs['bin']).reshape(-1,1), pandas_data), axis=1 )
        
    
    result_sin = sincerities( np.asarray(pandas_data ) )
    result_sin = result_sin[ 1:,1: ]
    if save:
        genes_subset = pd.read_csv('../../mouseprogenitormtxs/genes_for_subset.csv',index_col = 0)
        gene_names_to_subset = [str(idx) for idx in genes_subset['x'] ]
        result_pandas = pd.DataFrame(result_sin/result_sin.max() - np.diag( np.diag(result_sin/result_sin.max())) )
        
        result_pandas.index = gene_names_to_subset
        result_pandas.columns = gene_names_to_subset
        
        result_pandas.to_csv(saveName)

    return result_pandas

def subsample_and_concatenate(arr_list,  seed=42):
    np.random.seed(seed)  # Ensure reproducibility

    # Find the minimum number of columns
    min_cols = min(arr.shape[1] for arr in arr_list)
    for arr in arr_list:
        print( arr.shape[1] )
    print(min_cols)
    # Subsample each array to have 'min_cols' columns
    # subsampled_list = [arr[:, np.random.choice(arr.shape[1], min_cols, replace=False)] for arr in arr_list]
    subsampled_list = [arr[:, 0: min_cols] for arr in arr_list]
    
    # Concatenate along the column axis
    concatenated_array = np.hstack(subsampled_list)

    return subsampled_list, concatenated_array


def run_OTVelo( df, mode = 'Corr', time = 'days', save=False, saveName=None):
    import mygene

    mg = mygene.MyGeneInfo()
    # query all genes in your dataset
    mapping = mg.querymany(
        df.var_names.tolist(),
        scopes="ensembl.gene",
        fields="symbol",
        species="mouse"
    )
    
    # Convert to dictionary
    ensg2symbol = {entry["query"]: entry.get("symbol", None) for entry in mapping}
    df.var["gene_symbol"] = df.var_names.map(ensg2symbol)
    cell_cycle_genes = pd.read_csv('../../mouseprogenitormtxs/GO_cycle_genes_mouse.txt', sep=None, engine="python")
    genes_subset = list(set(list( cell_cycle_genes['MGI Gene/Marker ID'].values ) ) )
    gene_mask = df.var['gene_symbol'].isin(genes_subset)
    df_subset = df[:, gene_mask].copy()
    scanpy.tl.pca(df_subset, n_comps=5, svd_solver="arpack")  # or "randomized" for speed
    
    pandas_data = df.X.todense()
    
    if time == 'days':
        pandas_data = np.concatenate( ( np.array(df.obs['age (days)']).reshape(-1,1), pandas_data), axis=1 )
    elif time == 'tricycle':
        
        for i in range(df.obs.shape[0]):
            if ~np.isnan(df.obs['Tricycle'][i]):
                df.obs['bin'][i] = df.obs['tricycle-qcut'][i].mid
        pandas_data = np.concatenate( ( np.array(df.obs['bin']).reshape(-1,1), pandas_data), axis=1 )
    else:
        for i in range(df.obs.shape[0]):
            if ~np.isnan(df.obs['Tricycle'][i]):
                df.obs['bin'][i] = df.obs['theta-qcut'][i].mid
        pandas_data = np.concatenate( ( np.array(df.obs['bin']).reshape(-1,1), pandas_data), axis=1 )
    
    pandas_data = np.array( pandas_data.T )

    data_umap =df_subset.obsm['X_pca']
    
    counts_umap = np.array( data_umap.T )
    
    times = pandas_data[0,:].flatten()
    time_order = np.argsort( times )
    counts = pandas_data[1:, time_order]
    
    
    time_pts = np.sort(np.unique(times) )
    
    
    labels = np.zeros( (1,len(times)))
    for i in range(0,len(times)):
        # print(i)
        j = np.where( times[i] == time_pts)[0]
        # print(i)
        labels[0,i] = j
    Nt = len( np.unique(labels))
    counts_all = [ [0] ]*Nt
    counts_umap_all = [ [0] ]*Nt
    for i in range(Nt):
        idx = np.where( labels[0] == i )[0]
        counts_all[i] = counts[:, idx]
        counts_umap_all[i] = counts_umap[:, idx]
    dt = [0]*(Nt-1)
    for i in range(Nt-1):
        dt[i] = time_pts[i+1] - time_pts[i]

    if time == 'days':
        counts_all, counts = subsample_and_concatenate(counts_all, seed=42)
        counts_umap_all, counts_umap =  subsample_and_concatenate(counts_umap_all, seed=42)
        
        labels = np.zeros( (1, counts.shape[1] ))
        for j in range(len(time_pts)):
            # print(i)
            
            # print(i)
            labels[0,j*counts_all[0].shape[1]:(j+1)*counts_all[0].shape[1] ] = j
        print(counts.shape)
        print(Nt)
        print(labels)
    Ts_prior,_ = solve_prior(counts,counts, Nt, labels, eps_samp=1E-2, alpha=0.5)
    

    velocities_all, velocities_all_signed = solve_velocities( counts_all, Ts_prior, dt=dt)
    n = counts_all[0].shape[0]
    s = []
    for i in range(Nt):
        s = s + [counts_all[i].shape[1]]
    s_cum = list(np.cumsum(s))
    s_cum = [0] + s_cum
    
    velocities = np.zeros( (n, counts.shape[1]))
    velocities_signed = np.zeros( (n, (counts.shape[1])))
    
    for i in range(Nt):
        
        velocities[:,s_cum[i]:s_cum[i+1] ] = velocities_all[i]
        velocities_signed[:,s_cum[i]:s_cum[i+1]] = velocities_all_signed[i]

    if mode == 'Corr':
        Tv = OT_lagged_correlation(velocities_all_signed, velocities_signed, Ts_prior, return_slice=False)
    else:
        Tv = OT_lagged_correlation(velocities_all_signed, velocities_signed, Ts_prior, elastic_Net=True, l1_opt=0.5, signed=True, return_slice=False )

    if save:
        genes_subset = pd.read_csv('../../mouseprogenitormtxs/genes_for_subset.csv',index_col = 0)
        gene_names_to_subset = [str(idx) for idx in genes_subset['x'] ]
        result_pandas = pd.DataFrame(Tv/abs(Tv).max() - np.diag( np.diag(Tv/abs(Tv).max()) ) )
        result_pandas.index = gene_names_to_subset
        result_pandas.columns = gene_names_to_subset
        result_pandas.to_csv(saveName)
        
   



    return result_pandas

def matrix_to_top_regulations(csv_file, top_n=10, output_file=None):
    """
    Converts a square matrix of gene regulations to a list of the top regulations with source, target, and weight.
    
    Args:
        csv_file (str): Path to the CSV file containing the matrix. Rows and columns must represent gene names.
        top_n (int): Number of top regulations to return, sorted by absolute weight.
        output_file (str, optional): Path to save the resulting regulation list as a CSV. If None, the list is not saved.
        
    Returns:
        pd.DataFrame: A DataFrame containing the top regulations with columns 'Source', 'Target', and 'Weight'.
    """
    # Load the square matrix
    genes = pd.read_csv('../../mouseprogenitormtxs/10x_mouse_retina_development_feature.csv',index_col = 0)

    matrix = pd.read_csv(csv_file, index_col=0)
    
    
    print(np.max(matrix.abs() ))
    matrix = matrix/np.max(matrix.abs() )
    
    # Ensure the matrix is numeric
    matrix = matrix.apply(pd.to_numeric, errors='coerce')
    
    # Reshape the matrix into a long-format DataFrame
    matrix_long = matrix.stack().reset_index()
    matrix_long.columns = ['Source', 'Target', 'Weight']
    
    # Remove self-loops (if any)
    matrix_long = matrix_long[matrix_long['Source'] != matrix_long['Target']]
    
    # Sort by absolute weight in descending order
    matrix_long['AbsWeight'] = matrix_long['Weight'].abs()
    sorted_regulations = matrix_long.sort_values(by='AbsWeight', ascending=False)
    
    # Select the top N regulations
    top_regulations = sorted_regulations.head(top_n).drop(columns=['AbsWeight'])
    for i in range(top_n):
    
        top_regulations.iloc[i,0] = genes.loc[top_regulations.iloc[i].Source].gene_short_name
        top_regulations.iloc[i,1] = genes.loc[top_regulations.iloc[i].Target].gene_short_name
    # Save to a CSV file if an output file is specified
    if output_file:
        top_regulations.to_csv(output_file, index=False)
    
        
    return top_regulations
