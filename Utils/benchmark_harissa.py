import numpy as np
import pandas as pd
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
import pandas as pd
import numpy as np
import evaluation_metrics
from evaluation_metrics import evaluate_AUC, early_precision

import sys
sys.path.append('../Utils/')
from cycleGRN import bin_traj, train_ours,traj_ours,build_laplacian,find_Lie_derivative,time_lagged_correlation, time_lagged_Granger, co_moving_kernel_closed_form, pick_k_threshold, time_lagged_Granger_CV_fast,  perm_p_value


import scipy
import scanpy
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import pickle


def benchmark_HARISSA( dataset = 'CN5', cycle_genes=None, k=200, tau=0.95 ):

    for seed in range(1,11):
        data = pd.read_csv('/Users/wenjunzhao/dropbox/cardamom/results_article/Benchmark_on_simulated_data/CycleData/'+dataset+'/Data/data_'+str(seed)+'.txt',sep='\t',header=None)

        # t = data.iloc[0,201:]
        # data_cycle = data#data.iloc[[2,4,5],201:]


        data = data.iloc[1:,1:]
        if cycle_genes is None:
            data_cycle = data
            cycle_genes = list(range(data.shape[0]))
        else:
            data_cycle = data.iloc[cycle_genes,:]
        print(cycle_genes)

        import scanpy as sc
        adata = sc.AnnData(data.T)
        sc.pp.log1p(adata)

        adata_cycle = sc.AnnData(data_cycle.T)
        sc.pp.log1p(adata_cycle)

        if len(cycle_genes) > 2:
            sc.tl.pca(adata, n_comps=2)

            sc.tl.pca(adata_cycle, n_comps=2)

            X = adata_cycle.obsm["X_pca"]
        else:
            X = adata_cycle.X.todense()
        from cycleGRN import perm_p_value
        # _, p_adj, _ = perm_p_value(X, X, np.eye(X.shape[1]))
        
        # print(p_adj)
        # stop
        # Standardize the data
        # X_scaled = StandardScaler().fit_transform(X)

        # # PCA
        # pca = PCA(n_components=2)
        # pca_result = pca.fit_transform(X_scaled)
        pca_result = adata_cycle.obsm['X_pca'][:,0:2]

        # ys_true = (pca_result[:, :2] - pca_result[:, :2].min(0)) / np.ptp(pca_result[:, :2],0) * [8, 8] - [4, 4]
        ys_true = (pca_result[:, :2] - pca_result[:, :2].min(0)) / np.ptp(pca_result[:, :2]) * 8 - 4

        import pickle
        with open('../Data/temp.p', "wb") as f:
            pickle.dump([[0],ys_true], f)
        ys2,ts2,vs2 = traj_ours(500,'../Data/temp.p',seed=0)
        vs2 = vs2.detach().numpy()
        # k, tau = pick_k_threshold(ys_true, vs2, stimulation=True)
        # k = 200
        # tau = 0.95
        T = build_laplacian(ys_true, vs2, k = k, threshold=tau, stimulation=True)
        # T = co_moving_kernel_closed_form(ys_true, vs2.detach().numpy())
        X = adata.X.T

        V = find_Lie_derivative(X,T)

        G = time_lagged_correlation(X,V,T)
        _, p_adj, _ = perm_p_value(X, V, T)
        G_filter = G * (p_adj<0.2)
        # print(p_adj.shape)
        # print(G_filter)
        # stop
        # G_Granger,_ = time_lagged_Granger_CV_fast(X,V,T)
        #print(np.dot( np.mean( ys_true[0:100,:],axis=0) - np.mean(ys_true[100:200,:],axis=0), vs2[1,:] )  )
        # if np.dot( np.mean( ys_true[0:100,:],axis=0) - np.mean(ys_true[100:200,:],axis=0), vs2[1,:] ) < 0:
        #     G = G.T
        #     G_Granger = G_Granger.T
        #     print('reversed dynamics')
        if G[0,1] < G[1,0]:
            print('reversed dynamics')
            vs2 = -vs2
            T = build_laplacian(ys_true, vs2, k = k, threshold=tau, stimulation=True)
            # T = co_moving_kernel_closed_form(ys_true, vs2.detach().numpy())
            X = adata.X.T
    
            V = find_Lie_derivative(X,T)
    
            G = time_lagged_correlation(X,V,T)
            _, p_adj,_ = perm_p_value(X, V, T)
            G_filter = G * (p_adj<0.2)
            # G_Granger,_ = time_lagged_Granger_CV_fast(X,V,T)
            #print(np.dot( np.mean( ys_true[0:100,:],axis=0) - np.mean(ys_true[100:200,:],axis=0), vs2[1,:] )  )
            # if np.dot( np.mean( ys_true[0:100,:],axis=0) - np.mean(ys_true[100:200,:],axis=0), vs2[1,:] ) < 0:
            #     G = G.T
            #     G_Granger = G_Granger.T
            #     print('reversed dynamics')
        np.save('/Users/wenjunzhao/dropbox/cardamom/results_article/Benchmark_on_simulated_data/CycleData/'+dataset+'/CycleGRN/score_'+str(seed)+'.npy',G)
        # np.save('/Users/wenjunzhao/dropbox/cardamom/results_article/Benchmark_on_simulated_data/CycleData/'+dataset+'/CycleGRN_Granger/score_'+str(seed)+'.npy',G_Granger)

        np.save('/Users/wenjunzhao/dropbox/cardamom/results_article/Benchmark_on_simulated_data/CycleData/'+dataset+'/CycleGRN_filter/score_'+str(seed)+'.npy',G_filter)
        

def load_others_result(example, method, seeds, sign=None,undirected=False):
    AUROC = [0]*len(seeds)
    AUPRC = [0]*len(seeds)
    EP = [0]*len(seeds)
    #print( len( AUPRC) )
    counter = 0
    
    if example in ['FN4','CN5','BN8','FN8','NotchCycle']:
        # These datasets use the identical graph
        FileTruth =  '/Users/wenjunzhao/dropbox/cardamom/results_article/Benchmark_on_simulated_data/CycleData/'+example+'/True/inter_signed.npy'
        Tv_true = np.load(FileTruth)
        
        Tv_true = Tv_true - np.diag( np.diag(Tv_true) )
        
        
        
    
    
    for s in seeds:
        
            
            
        FileResult = '/Users/wenjunzhao/dropbox/cardamom/results_article/Benchmark_on_simulated_data/CycleData/'+example+'/'+method+'/score_'+str(s)+'.npy'
               #print( np.load(FileResult) )
        

        Tv_total = np.load(FileResult)
        
        #print(Tv_total.shape[0])
        
        if method == 'GENIE3':
            Tv_total = Tv_total.T
        
        
        Tv_total = Tv_total - np.diag( np.diag( Tv_total ))

        # if abs(Tv_total[1,0]) > abs(Tv_total[0,1]):
        #     Tv_total = Tv_total.T
        
        n = Tv_total.shape[0]
        Tv_total_flattened = []
        Tv_true_flattened = []

        for i in range(n):
            for j in range(n):
                if i != j:
                    Tv_total_flattened += [Tv_total[i,j]]
                    Tv_true_flattened += [Tv_true[i,j]]
        Tv_total_flattened = np.array( Tv_total_flattened )
        Tv_true_flattened = np.array(Tv_true_flattened)
        

        
        
        
        
        from sklearn.metrics import roc_curve, precision_recall_curve, auc, roc_auc_score
        if undirected == True:
            
            Tv_true = abs(Tv_true) + abs(Tv_true).T
            Tv_true = Tv_true>0
            Tv_total_undirected = np.zeros((n,n))
            for i in range(n):
                for j in range(n):
                    Tv_total_undirected[i,j] = max( abs(Tv_total[i,j]),abs(Tv_total[j,i]) )
            AUPRC[counter], AUROC[counter], random = evaluate_AUC( Tv_total_undirected, Tv_true, sign=None)
        else:
            AUPRC[counter], AUROC[counter], random = evaluate_AUC( Tv_total, Tv_true, sign = sign )
            # precision, recall, thresholds = precision_recall_curve(abs(Tv_true_flattened), abs(Tv_total_flattened) )
            # AUPRC[counter] = auc(recall,precision)
        EP[counter] = early_precision( Tv_total, Tv_true, sign=sign)
        
        counter += 1
        
    return AUPRC, AUROC, EP, random