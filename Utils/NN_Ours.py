from tqdm.notebook import tqdm  # Use 'from tqdm import tqdm' if running in a script/terminal

def traj_ours(TIME,name,seed,lr=1e-3,plot=False):#input the desired wall clock training time and the name of the trajectory file to learn
    
    
    ###########################################################
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from mpl_toolkits.mplot3d import Axes3D
    from time import time
    import numpy as np
    import numpy.ma as ma
    import matplotlib.pyplot as plt
    import copy
    from scipy.sparse import lil_matrix, csr_matrix, identity,spdiags, linalg
    import scipy.sparse as sparse
    from scipy.sparse.linalg import LinearOperator,eigs,spsolve,norm
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from scipy.interpolate import RegularGridInterpolator
    from scipy.interpolate import RBFInterpolator
    import matplotlib.animation as animation
    from time import time
    from numpy.random import default_rng
    import math 
    import torch
    import torch.nn.functional as F
    import torch.nn as nn
    from torch.distributions import normal
    import torch.optim as optim
    import matplotlib.pylab as plt
    from IPython import get_ipython
    from scipy.ndimage import gaussian_filter
    import pickle
    import sys
    import torch.optim as optim
    import os
    import json
    from datetime import datetime
    from scipy.optimize import minimize, LinearConstraint
    ##########################################################


    #Load in binned trajectory data
    with open(name, "rb") as f:
        data = pickle.load(f)
        
    ys = data[1]
    G = {
    "dt": .01, #timestep
    "dx": .1, #spatial discretization
    'bounds': [[-4,4],[-4,4]], #bounds used in discretization
    "alpha": 1e-8, #teleportation parameter
    "diff": 1e-3, #diffusion parameter
    'Cost': 'KL',#cost function, can be L2,W2,KL
    'nodes': 100, #number of nodes in NN
    'act': 'tanh', #NN activation
    'lr': lr, #learning rate
    'TSMax': 1e6, #number of steps for plotting dynamics
    'filtering': 2, #standard deviation of gaussian kernel for filtering
    'numiter': TIME,#00000, #total number of iterations
    'plotevery': 1000,#0000, #how often to plot updates (no plots saved if > numiters)
    'tol': .0001, #stopping tolerance
    'end': False,
    'plot': plot,
    }
    ########################################################################
    
    
    bounds = G['bounds']
    
    Xi=[i for i in np.arange(bounds[0][0], bounds[0][1]+G["dx"],G["dx"])]
    Yi=[i for i in np.arange(bounds[1][0], bounds[1][1]+G["dx"],G["dx"])]
    
    Xf=[i-G["dx"]/2 for i in np.arange(bounds[0][0], bounds[0][1]+ G["dx"] + G["dx"] ,G["dx"])]
    Yf=[i-G["dx"]/2 for i in np.arange(bounds[1][0], bounds[1][1]+ G["dx"] + G["dx"] ,G["dx"])]
    
    G['nx'] = len(Xi)
    G['ny']  = len(Yi)
    
    
    Xi_int=[i for i in np.arange(bounds[0][0], bounds[0][1]+2*G["dx"],G["dx"])]
    Yi_int=[i for i in np.arange(bounds[1][0], bounds[1][1]+2*G["dx"],G["dx"])]
    
    Xi = np.array(Xi)
    Yi = np.array(Yi)
    
    xv, yv  = np.meshgrid(Xi, Yi, sparse=False, indexing='ij')
    
    X = np.zeros((G['nx']*G['ny'],2))
    X[:,0] = xv.reshape(G['nx']*G['ny'], order='F')
    X[:,1] = yv.reshape(G['nx']*G['ny'], order='F')
    
    
    hist_bounds = [[Xf[0],Xf[-1]],[Yf[0],Yf[-1]]]
    Peq_true, edges = np.histogramdd(ys , range = hist_bounds, bins = [G['nx'],G['ny']],density=True)
    Peq_true = Peq_true/sum(Peq_true.flatten())
    from scipy.ndimage import gaussian_filter
    Peq_true = gaussian_filter(Peq_true,sigma =  G['filtering'])
    if G['plot']:
        plt.imshow(Peq_true.T,origin = 'lower',aspect = 'auto')
        
        plt.show()
    Peq_true = Peq_true.flatten(order = 'F')
    #cost matrix Wasserstein metric (W2)
    
    nx = G['nx']
    ny = G['ny']
    
    
    DxR,DxL,DxC = np.ones((nx,ny)),np.ones((nx,ny)),-2*np.ones((nx,ny))
    DyR,DyL,DyC = np.ones((nx,ny)),np.ones((nx,ny)),-2*np.ones((nx,ny))
    DxR[-1,:],DxL[0,:],DyR[:,-1], DyL[:,0] = 0,0,0,0
    DxC[0,:],DxC[-1,:], DyC[:,0],DyC[:,-1] = -1,-1,-1,-1
    DxR,DxL,DxC,DyR,DyL,DyC = G['diff']*DxR.flatten(order = 'F'),G['diff']*DxL.flatten(order = 'F'),G['diff']*DxC.flatten(order = 'F'),G['diff']*DyR.flatten(order = 'F'),G['diff']*DyL.flatten(order = 'F'),G['diff']*DyC.flatten(order = 'F')

    ###########Solve Forward Problem
    def diffmin(x):
        return 1.*(x<0)
    def diffmax(x):
        return 1.*(x>0)
    
    def RHS_Matrix2 (G,bounds):
        Uf = G['Vx']
        Vf = G['Vy']
        up = np.maximum(Uf,0)
        un = np.minimum(Uf,0)
        vp = np.maximum(Vf,0)
        vn = np.minimum(Vf,0)
        #Build sparse time advance operator K_mat
        #Each row (first index) is the equation for one [i,j,k] cell in terms of +-1 neighbors
        N=nx*ny
        
        # This is the equivalent of "ndgrid" in Matlab by specifing indexing='ij'  
        XXf,YYf= np.meshgrid(Xf,Yf,indexing='ij')
       
        # ufm=Uf[i-1,j-1,k-1]
        ufm = Uf[:-1,:-1]
        ufm = ufm.flatten(order='F')
      
    
        #  ufp=Uf[i,j-1,k-1]
        ufp = Uf[1:,:-1]
        ufp = ufp.flatten(order='F')
      
                   
        # vfm=Vf[i-1,j-1,k-1]
        vfm = Vf[:-1,:-1]
        vfm = vfm.flatten(order='F')
       
    
        # vfp=Vf[i-1,j,k-1]
        vfp = Vf[:-1,1:]
        vfp = vfp.flatten(order='F')
        
        T1 = spdiags(np.array([np.maximum(0,ufp)+DxR/G['dx'], np.minimum(0,ufm)- np.maximum(0,ufp)+DxC/G['dx'], -np.minimum(0,ufm)+DxL/G['dx']]), np.array([-1,0,1]), N, N) 
        T2 = spdiags(np.array([np.maximum(0,vfp)+DyR/G['dx'], np.minimum(0,vfm)- np.maximum(0,vfp)+DyC/G['dx'], -np.minimum(0,vfm)+DyL/G['dx']]), np.array([-nx,0,nx]), N, N)
       
        K_mat = (T1 + T2) /G['dx']*G['dt']
        return  K_mat
    
    
    def FWD(G,bounds):  
        K_mat = RHS_Matrix2(G,bounds)
        K_mat = K_mat.tocsr()
    
        # scale to positive
        mnV = abs(K_mat.min(axis=0).min())  
        # print(mnV)
        N = K_mat.shape[0]
        speyeN = identity(N).tocsr()
        M = speyeN +(1/(2*mnV))*K_mat
        e = np.ones(N,)
    
        A = (1-G["alpha"])*M - speyeN
        b = -(G["alpha"]/N)*e
    
        x = spsolve(A, b )
        Peq = x.reshape((G['nx'],G['ny']), order='F')
       # print(np.sum(x))
    
        ##### Reshape the solution
        Peq0 = Peq * 0
        Peq0[1:-1, 1:-1] = Peq[1:-1, 1:-1]
        Peq0 = Peq0/Peq0.sum()
        Peq0 = Peq0.flatten(order = 'F')
        
    
        # return the transpose of matrix (1-G["alpha"])*M - speyeN for adjoint eqn.
        return (1-G["alpha"])*M.transpose() - speyeN, Peq0, x, K_mat, M
    
    
    def safe_divide(n, d):    
        return n / d if d else 0
    
    
    
    def KL(n,d):
        if n!=0 and d!= 0:
            return n*np.log(n/d)
        else:
            return 0
        
    def KLd(n,d):
        if n!=0 and d!=0:
            return -n/d
        else:
            return 0
        
    def JS(n,d):
        if n!=0 and d!= 0:
            m = (n+d)/2
            return (KL(n,m)+KL(d,m))/2
        else:
            return 0
    
    def JSd(n,d):
        if n!=0 and d!= 0:
            return .5*np.log((2*d)/(n+d))
        else:
            return 0
    
    
   
    
   
    
    
    #####Calculate Gradient
    def calc_cost_gradient(G,bounds,Peq_true):
        #### 1. Solve the forward problem
        A,Peq,_,K_mat,_ = FWD(G,bounds)
        #### 2. compute the loss function
    
        if G['Cost'] == 'KL':
            cost = np.sum([KL(Peq_true[i],Peq[i]) for i in range(len(Peq))])
            u = np.asarray([KLd(Peq_true[i],Peq[i]) for i in range(len(Peq))])
        if G['Cost'] == 'JS':
            cost = np.sum([JS(Peq_true[i],Peq[i]) for i in range(len(Peq))])
            u = np.asarray([JSd(Peq_true[i],Peq[i]) for i in range(len(Peq))]) 
        if G['Cost'] == 'L2':
            cost = np.linalg.norm(Peq - Peq_true)**2 *0.5 
            u = Peq-Peq_true 
        #### 3. Solve the adjoint equation
        sol = spsolve(A,-u + u.dot(Peq))
        #### 4. compute the gradient
        grad1, grad2 = np.zeros((nx+1,ny+1)),np.zeros((nx+1,ny+1))
        mnV = abs(K_mat.min(axis=0).min())
        for i in range(1,nx-1):
          for j in range(1,ny-1):
            idx = i + nx*j
            dv1, dv2 = 1,1
            v1, v2 = G['Vx'][i,j], G['Vy'][i,j]
            grad1[i,j] = ((G['dt']*(1-G["alpha"])*(1/(2*mnV))*(sol[idx]-sol[idx-1])*(diffmax(v1)*Peq[idx-1]+diffmin(v1)*Peq[idx])*dv1/G['dx'])) 
            grad2[i,j] = ((G['dt']*(1-G["alpha"])*(1/(2*mnV))*(sol[idx]-sol[idx-nx])*(diffmax(v2)*Peq[idx-nx]+diffmin(v2)*Peq[idx])*dv2/G['dx'])) 
        
        
        g1 = grad1[1:-1,1:-1]
        g2 = grad2[1:-1,1:-1]
        
        return cost,g1.flatten(),g2.flatten()
    
    Iterations = 1
    iters = []
    costs = []
    
    
    def interp_col(y,C): #given a vector y in R3, return interpolated parameter values
        interp = RegularGridInterpolator((Xi, Yi), C, method= 'linear', bounds_error=False,fill_value=None)
        return interp(y)
    
    def dynamics(G):   
        def RHS_NN(x,G):    
            v = net(torch.tensor(x,dtype = torch.float).reshape(2,1)).detach().numpy()[0]
            return np.array([v[0],v[1]])
        
    
        
        print('Plotting Dynamics')
        Vx, Vy = G['Vx'].copy(), G['Vy'].copy()
      
       
    
        TimeStepMax = G['TSMax']
        IC = G['IC']
        ys = np.zeros((int(TimeStepMax),2))
        ys[0,:] = ys[0,:] = IC
        ysn = ys.copy()
        # tic = time()
        # noiseV = np.random.normal(0,1,int(2*TimeStepMax)).reshape((int(TimeStepMax),2))  
        for timestep in range(1,int(TimeStepMax)): 
            # if timestep% int(TimeStepMax/10)==0:
               # toc = time()
              # print(f'time step: {timestep:4d} Elapsed time: {toc-tic:.2f}s')
            ## https://en.wikipedia.org/wiki/Euler%E2%80%93Maruyama_method
            # ys[timestep,:] = ys[timestep-1,:] + G["dt"]* RHS_NN(ys[timestep-1,:],G) +np.sqrt(2*G['diff']*G['dt'])*noiseV[timestep-1,:]
            ysn[timestep,:] = ysn[timestep-1,:] + G["dt"]* RHS_NN(ysn[timestep-1,:],G)  
        
        G['Vx'], G['Vy'] = Vx, Vy
        
    
        return ysn
    
    
    def velocities(G):
        print('Plotting Inferred Parameter')
        Vx, Vy = G['Vx'].copy(), G['Vy'].copy()
      
        Np = 300
        Xp = np.linspace(bounds[0][0]+G['dx'],bounds[0][1]-G['dx'],Np)
        Yp = np.linspace(bounds[1][0]+G['dx'],bounds[1][1]-G['dx'],Np)
        # Np = nx
        # Xp = Xi
        # Yp = Yi
        
        
        P = np.zeros((Np,Np,2)) 
        P[:,:,0],P[:,:,1] = np.meshgrid(Xp,Yp,indexing = 'ij')
        P = np.zeros((Np,Np,2)) 
        P[:,:,0],P[:,:,1] = np.meshgrid(Xp,Yp,indexing = 'ij')
      #y_big is n by 3 where n= (nx-1)*(ny-1)*(nz-1) is # of points we evaluate the RHS
        P = P.reshape((Np**2,2)).T
        Vi = net(torch.tensor(P,dtype = torch.float)).detach().numpy()
        V1 = Vi[:,0].reshape((Np,Np))
        V2 = Vi[:,1].reshape((Np,Np))
        G['Vx'], G['Vy'] = Vx, Vy
        if G['plot']:
            plt.streamplot(Xp,Yp,V1.T,V2.T)
            plt.show()
        return V1,V2
    
    def velocities_in_sample(G,ys):
        vs = torch.zeros_like(torch.tensor(ys,dtype=torch.float) )
        for i in range(ys.shape[0]):
            vs[i,:] = net(torch.tensor(ys[i,:],dtype=torch.float) ).detach()#.numpy()
        return vs#net(  torch.tensor(ys,dtype = torch.float)).detach().numpy() 
    
    
    
    relu = nn.ReLU()
    sig = nn.Sigmoid()
    
    def activation(z):
        if G['act'] == 'relu':
            return relu(z)
        if G['act'] == 'sig':
            return sig(z)
        if G['act'] == 'tanh':
            return torch.tanh(z)
    
    
    #Network for V
    class network(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(2, 100),
                nn.Tanh(),
                nn.Linear(100, 2),
            )
        def forward(self, y):
            return self.net(y.T)
    
    class EarlyStopper:
        def __init__(self, patience=50, min_delta=1e-5):
            """
            patience: How many iterations to wait after the last improvement.
            min_delta: Minimum change in loss to qualify as an improvement.
            """
            self.patience = patience
            self.min_delta = min_delta
            self.counter = 0
            self.best_loss = np.inf
            self.early_stop = False
    
        def __call__(self, val_loss):
            if val_loss < (self.best_loss - self.min_delta):
                self.best_loss = val_loss
                self.counter = 0  # Reset counter if we improved
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
    
        
    class PDEsolver(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            """
            Forward equation
            """
            V = input.detach().numpy() # input has output from NN evaluated at the mesh
    
            G['Vx'][1:-1,1:-1] = V[:,0].reshape((nx-1,ny-1))
            G['Vy'][1:-1,1:-1] = V[:,1].reshape((nx-1,ny-1))
            g = np.zeros((2,(nx-1)*(ny-1)))
            c,g[0,:],g[1,:] = calc_cost_gradient(G,bounds,Peq_true)
    
        
    
            ctx.save_for_backward(torch.tensor(g.T, dtype = torch.float)) # save something for backward in NN
    
            return torch.tensor(c, dtype = torch.float)
        @staticmethod
        def backward(ctx, grad_output):
            g, = ctx.saved_tensors  # pull it out from what is saved earlier ctx is like "self"
            return g, None
    
    
    
    XX = torch.tensor(np.meshgrid(Xf[1:-1],Yi_int[1:-1],indexing='ij'), dtype = torch.float)
    XX = XX.reshape((2,len(Xf[1:-1])*len(Yi_int[1:-1])))
    
    YY = torch.tensor(np.meshgrid(Xi_int[1:-1],Yf[1:-1],indexing='ij'), dtype = torch.float)
    YY = YY.reshape((2,len(Xi_int[1:-1])*len(Yf[1:-1])))
    
    torch.manual_seed(seed)
    net = network()
    Vi = net(XX).detach().numpy()
    Vj = net(YY).detach().numpy()
    G['Vx'], G['Vy'] = np.zeros((nx+1,ny+1)),np.zeros((nx+1,ny+1))
    
    def bin_traj(ys):
        Peq, _ = np.histogramdd(ys , range = [[-4,4],[-4,4]], bins = [50,50], density=True)
        Peq = Peq/sum(Peq.flatten())
        return Peq
    
    _,Peq_initial,_,_,_ = FWD(G,bounds)
    
    solver = PDEsolver.apply
    
    steps = G['numiter']
    
    loss_history = []
    grad_history = []
    reg_history = []
    total_history = []
    
    
    # opt = torch.optim.Adam(net.parameters(), lr=G['lr'])
    # start = time()
    # for k in range(steps):
    #     net.train()
            
    #     def closure1():
    #         opt.zero_grad() 
    #         y_pred = torch.zeros((nx-1)*(ny-1),2)
    #         y_pred[:,0] = net(XX)[:,0]
    #         y_pred[:,1] = net(YY)[:,1]
    #         loss = solver(y_pred)
    #         loss.backward()
    #         return loss
    
    #     loss = opt.step(closure1)
       
        
    
    #     #Update velocities by NN
            
    #     Vi = net(XX).detach().numpy()
    #     Vj = net(YY).detach().numpy()
        
    #     G['Vx'][1:-1,1:-1] = Vi[:,0].reshape((nx-1,ny-1))
    #     G['Vy'][1:-1,1:-1] = Vj[:,1].reshape((nx-1,ny-1))
    #     # G['Vx'], G['Vy'] = np.zeros((nx+1,ny+1)),np.zeros((nx+1,ny+1))
    
    
        
    
    
    #     # print('Cost: ', loss.detach().numpy())
    #     costs.append(loss.detach().numpy())
    #     iters.append(k)
    
                
    #     if k%G['plotevery']  == 0 and G['plot']:
    #         print('Iteration', k,'|', 'Cost:',loss.detach().numpy(), '|', 'Tol:', costs[-1]/costs[0])
          
    
    #     if k%G['plotevery'] == 0 and k!= 0 and G['plot']:
    #         _,Peq,_,_,M = FWD(G,bounds)
    #         plt.imshow(Peq.reshape(nx,ny,order = 'F').T,origin = 'lower',aspect = 'auto')
    #         plt.show()
    #         velocities(G)
    #         plt.show()
    #     end = time()
    #     if end-start > TIME:
    #         break
    #     k+=1
    # Initialize optimizer
    early_stopper = EarlyStopper(patience=100, min_delta=1e-5) 

    opt = torch.optim.Adam(net.parameters(), lr=G['lr'])
    pbar = tqdm(range(steps), desc="Training CYCLO")
    start = time()
    
    for k in pbar:
        net.train()
            
        def closure1():
            opt.zero_grad() 
            y_pred = torch.zeros((nx-1)*(ny-1), 2)
            y_pred[:,0] = net(XX)[:,0]
            y_pred[:,1] = net(YY)[:,1]
            loss = solver(y_pred)
            loss.backward()
            return loss
        
        loss = opt.step(closure1)
        current_loss = loss.item()
        
        # Update progress bar
        pbar.set_postfix({'Cost': f'{current_loss:.5f}'})
    
        # --- 3. Check Early Stopping ---
        early_stopper(current_loss)
        if early_stopper.early_stop:
            print(f"Early stopping triggered at iteration {k}. Loss stabilized at {current_loss:.6f}")
            break  # <--- Exits the loop immediately
    
        # Update velocities by NN
        Vi = net(XX).detach().numpy()
        Vj = net(YY).detach().numpy()
        
        G['Vx'][1:-1,1:-1] = Vi[:,0].reshape((nx-1,ny-1))
        G['Vy'][1:-1,1:-1] = Vj[:,1].reshape((nx-1,ny-1))
    
        costs.append(current_loss)
        iters.append(k)
        
        # Plotting logic
        if k % G['plotevery'] == 0 and k != 0 and G['plot']:
            _, Peq, _, _, M = FWD(G, bounds)
            plt.figure(figsize=(10,4))
            plt.subplot(1,2,1)
            plt.imshow(Peq.reshape(nx,ny,order='F').T, origin='lower', aspect='auto')
            plt.title(f"Iteration {k}")
            plt.show()
            velocities(G)
            plt.show()
    
        # Time check break
        end = time()
        if end - start > TIME:
            print("Time limit reached.")
            break
            

    # end = time()
    G['dt'] = .01
    G['IC'] = np.array([ 0.84144155, -1.08920043])
    ysn = dynamics(G)
    if G['plot']:
        plt.scatter(ysn[:,0],ysn[:,1],c = 'r')
    _,Peq,_,_,M = FWD(G,bounds)
    v1,v2 = velocities(G)
    Ms = project_M_to_samples_operator(M, ys)
    Ms = linear_operator_to_dense(Ms)
    return ysn, end-start,velocities_in_sample(G,ys)
     
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator

def build_W_bilinear(ys, xmin=-4.0, xmax=4.0, ymin=-4.0, ymax=4.0, dx=0.1):
    """
    Build sparse bilinear interpolation matrix W (N x n) that maps sample-mass -> grid-mass.

    If s in R^n is a distribution over samples (sum=1),
      p_grid = W @ s  is the induced distribution over grid nodes (sum=1)
    using bilinear weights to distribute each sample to its 4 surrounding grid nodes.

    Convention:
      Grid has nodes at xmin + i*dx, i=0..nx-1; similarly for y.
      Flattening is Fortran: idx = i + nx*j.
    """
    ys = np.asarray(ys, dtype=float)
    n = ys.shape[0]

    # grid sizes (include endpoints)
    nx = int(round((xmax - xmin) / dx)) + 1
    ny = int(round((ymax - ymin) / dx)) + 1
    N = nx * ny

    x = ys[:, 0]
    y = ys[:, 1]

    # Clamp points to the grid interior so we can take (i0,i0+1) safely
    # Points exactly at xmax/ymax go to the last cell (nx-2, ny-2)
    eps = 1e-12
    x = np.clip(x, xmin, xmax - eps)
    y = np.clip(y, ymin, ymax - eps)

    # fractional index in grid coordinates
    fx = (x - xmin) / dx
    fy = (y - ymin) / dx

    i0 = np.floor(fx).astype(int)
    j0 = np.floor(fy).astype(int)
    i1 = i0 + 1
    j1 = j0 + 1

    # Safety
    i0 = np.clip(i0, 0, nx - 2)
    i1 = np.clip(i1, 1, nx - 1)
    j0 = np.clip(j0, 0, ny - 2)
    j1 = np.clip(j1, 1, ny - 1)

    tx = fx - i0
    ty = fy - j0

    # Bilinear weights
    w00 = (1 - tx) * (1 - ty)
    w10 = tx * (1 - ty)
    w01 = (1 - tx) * ty
    w11 = tx * ty

    # Convert (i,j) to flattened grid indices (Fortran order)
    def idx(ii, jj):
        return ii + nx * jj

    g00 = idx(i0, j0)
    g10 = idx(i1, j0)
    g01 = idx(i0, j1)
    g11 = idx(i1, j1)

    # Build sparse W: each column (sample) has 4 nonzeros summing to 1
    rows = np.concatenate([g00, g10, g01, g11])
    cols = np.concatenate([np.arange(n), np.arange(n), np.arange(n), np.arange(n)])
    data = np.concatenate([w00, w10, w01, w11])

    W = sp.csr_matrix((data, (rows, cols)), shape=(N, n))
    return W, (nx, ny)
   
def project_M_to_samples_operator(M, ys, dx=0.1, xmin=-4, xmax=4, ymin=-4, ymax=4):
    """
    Returns (P_op, W) where P_op acts like P = W^T M W on vectors in R^n,
    without forming P explicitly.
    """
    W, _ = build_W_bilinear(ys, xmin, xmax, ymin, ymax, dx)
    n = ys.shape[0]

    def matvec(v):
        v = np.asarray(v, dtype=float).ravel()
        return W.T @ (M @ (W @ v))

    def rmatvec(v):
        v = np.asarray(v, dtype=float).ravel()
        # (W^T M W)^T = W^T M^T W
        return W.T @ (M.T @ (W @ v))

    P_op = LinearOperator(shape=(n, n), matvec=matvec, rmatvec=rmatvec, dtype=float)
    return P_op

import numpy as np

def linear_operator_to_dense(Ms):
    n, m = Ms.shape
    assert n == m, "Operator must be square"

    A = np.zeros((n, n))
    for j in range(n):
        e = np.zeros(n)
        e[j] = 1.0
        A[:, j] = Ms @ e
    return A
    
