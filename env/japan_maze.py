import gym
import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
class JapanMaze(object):
    def __init__(self,radius=0.5,seed=0):
        np.random.seed(seed=seed)
        self.action_limit = 0.1
        self.ini_posi = np.array([-0.9,-0.9])
        self.ini_cov = np.array([[0.005,0.],[0.,0.005]])
        self.whereami = copy.deepcopy(self.ini_posi)
        self.goal = np.array([0.9,0.9])
        self.reward_f = lambda y:np.exp(-(np.linalg.norm(y-self.goal)**2)/2) 
        self.center = np.array([0.0,0.0])
        self.radius = radius 
        self.timelimit =40
        self.N = 30 # Collision determination　resolution
        high = np.ones(2)*1
        self.observation_space = gym.spaces.Box(low=-np.ones(2)*1, high=np.ones(2)*1,dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-np.ones(2)*0.1, high=np.ones(2)*0.1,dtype=np.float32)

    def reset(self):
        self.timestep = 0
        self.whereami = np.random.multivariate_normal(self.ini_posi, self.ini_cov)
        return self.whereami
    
    def isvalid(self,wai):
        return np.linalg.norm(self.center-wai) >= 0.5

    def step_near_circle(self,ac):
        wai = copy.deepcopy(self.whereami)
        for i in range(1,self.N+1):
            ratio = i/self.N
            n_wai = self.whereami+ac*ratio
            if not self.isvalid(n_wai): # 丸の中入ったら，一個前を返す
                return wai
            else:
                wai = copy.deepcopy(n_wai)
        return wai
        
    def step(self,ac):
        self.timestep +=1
        ac = copy.deepcopy(np.array([max(-self.action_limit,min(ac[0],self.action_limit)),
                           max(-self.action_limit,min(ac[1],self.action_limit))]))
        ac += np.random.normal(0.0, 0.005, 2)
        nwai = self.whereami+ ac
        nwai[0] = min(max(-1.,nwai[0]),1.)
        nwai[1] = min(max(-1.,nwai[1]),1.)
        if nwai[0] < 0.5 and nwai[0] > -0.5 and nwai[1] < 0.5 and nwai[1] > -0.5:
            self.whereami = self.step_near_circle(ac)
        else:
            self.whereami = nwai
        rew = self.reward_f(self.whereami)
        return self.whereami, rew, self.timestep>=self.timelimit,{}

    def render(self):
        fig = plt.figure(figsize=(5,5))
        ax = plt.axes()
        ax.plot([-1,-1,1,1,-1],[-1,1,1,-1,-1],c='black')
        c = patches.Circle(xy=(self.center[0], self.center[1]), radius=self.radius, fc='r', ec='r')
        ax.add_patch(c)
        ax.scatter(self.whereami[0],self.whereami[1],c='black')
        ax.scatter(self.goal[0],self.goal[1],marker='x',c='black')

    def vis_trace(self,X,Y,save_name=False):
        fig = plt.figure(figsize=(5,5))
        ax = plt.axes()
        ax.plot([-1,-1,1,1,-1],[-1,1,1,-1,-1],c='black')
        c = patches.Circle(xy=(self.center[0], self.center[1]), radius=self.radius, fc='r', ec='r')
        ax.add_patch(c)
        for x,y in zip(X,Y):
            if  y[0] == 0 and y[1] == 0:
                ax.scatter(x[0],x[1],c='black',s=5)
            else:
                ax.arrow(x=x[0],y=x[1],dx=y[0],dy=y[1],
                         width=0.002,head_width=0.05,
                         head_length=0.02,
                         length_includes_head=True,color='k')
        ax.scatter(self.goal[0],self.goal[1],marker='x',c='black')
        if save_name:
            plt.savefig(save_name,format = 'png', dpi=200)

    def vis_gpr(self,pilco,save_name=False):
        posi = [[-0.7,-0.7],[0.7,-0.7],[-0.7,0.7],[0.7,0.7]]
        th = lambda t:[np.cos(t)*0.1,np.sin(t)*0.1]
        vec = [th(np.pi/2),th(np.pi/2 + 2/3*np.pi),th(np.pi/2 + 4/3*np.pi)]
        posi_vec = np.array([p+v for p in posi for v in vec])
        xmean,xvar = pilco.mgpr.models[0].predict_y(posi_vec)
        ymean,yvar = pilco.mgpr.models[1].predict_y(posi_vec)
        means = np.hstack((xmean,ymean))
        std = np.sqrt(np.hstack((xvar,yvar)))
        # get ready lets draw!
        fig = plt.figure(figsize=(10,10))
        ax = plt.axes()
        ax.plot([-1,-1,1,1,-1],[-1,1,1,-1,-1],c='black')
        c = patches.Circle(xy=(self.center[0], self.center[1]), radius=self.radius, fc='r', ec='r')
        ax.add_patch(c)
        for i in range(len(ymean)):
            ax.arrow(x=posi_vec[i][0],y=posi_vec[i][1],dx=posi_vec[i][2],dy=posi_vec[i][3],
                 width=0.001,head_width=0.01,
                 head_length=0.01,
                 length_includes_head=True,color='k')
            errorx = means[i][0]+posi_vec[i][:2][0]-(posi_vec[i][0]+posi_vec[i][2])
            errory = means[i][1]+posi_vec[i][:2][1]-(posi_vec[i][1]+posi_vec[i][3])
            ax.arrow(x=posi_vec[i][0]+posi_vec[i][2],y=posi_vec[i][1]+posi_vec[i][3],
                     dx=errorx,dy=errory,width=0.001,head_width=0.01,head_length=0.01,
                     length_includes_head=True,color='red')
            e2 = patches.Ellipse(xy = means[i]+posi_vec[i][:2], width = std[i][0], height = std[i][1], 
                 alpha = 1.0, ec = "red", fill=False)
            ax.add_patch(e2)
        if save_name:
            plt.savefig(save_name,format = 'png', dpi=200)


    def vis_policy(self,pilco,save_name=False):
        pol = lambda x:pilco.compute_action(x[None, :])[0, :]
        centers = [np.array([i/10-1,j/10-1]) for i in range(0,21) for j in range(0,21)]
        vecs = [pol(c) for c in centers]

        fig = plt.figure(figsize=(10,10))
        ax = plt.axes()
        ax.plot([-1,-1,1,1,-1],[-1,1,1,-1,-1],c='black')
        c = patches.Circle(xy=(self.center[0], self.center[1]), radius=self.radius, fc='r', ec='r')
        ax.add_patch(c)
        for cent,v in zip(centers,vecs):
            ax.arrow(x=cent[0],y=cent[1],dx=v[0],dy=v[1],
                 width=0.002,head_width=0.03,
                 head_length=0.02,
                 length_includes_head=False,color='k')
        ax.scatter(self.goal[0],self.goal[1],marker='x',c='black')
        if save_name:
            plt.savefig(save_name,format = 'png', dpi=200)
        
    def vis_reward(self):
        fig, ax = plt.subplots(1, 2, figsize=(13, 5))
        # colormap
        rew = [self.reward_f([i/100-1,j/100-1]) for i in range(200) for j in range(200)]
        x = np.array([[i/100-1,j/100-1] for i in range(200) for j in range(200)])
        c = patches.Circle(xy=(self.center[0], self.center[1]), radius=self.radius, fc='r', ec='r')
        ax[1].add_patch(c)
        ax[1].plot([-1,-1,1,1,-1],[-1,1,1,-1,-1],c='black')
        ax[1].plot([-1,1],[-1,1], ':',c='black', alpha=0.7)
        im = ax[1].scatter(x[:,0],x[:,1],c=rew)
        ax[1].scatter(self.goal[0],self.goal[1],marker='x',c='black')
        fig.colorbar(im)
        # Cross section
        rews = [self.reward_f([i/100-1,i/100-1]) for i in range(200)]
        ax[0].plot([j/100-1 for j in range(200)],rews, ':', c='black')
        ax[0].set_xlabel('x=y=')
        ax[0].set_ylabel('reward')
        # ax[0].vlines([self.goal[0]], min(rews), max(rews), "black", linestyles='dashed')
        ax[0].scatter(self.goal[0],max(rews),marker='x',c='black')

    def vis_kernel2d(self):
        other=np.array([0.9,0.9, 0.1, 0.0])
        import gpflow
        base_vec = np.array([0.1,0.])
        fig, ax = plt.subplots(1, 3, figsize=(20, 5))
        k = gpflow.kernels.RBF(input_dim=4, variance=1., ARD=True)
        calk = lambda xy,vec:(k.compute_K(np.array([[xy[0],xy[1], vec[0], vec[1]]]), 
                      other.reshape(1,4))[0][0])
        x = np.array([[i/50-1,j/50-1] for i in range(100) for j in range(100)])
        k_val = np.array([calk([i/50-1,j/50-1],base_vec) for i in range(100) for j in range(100)])
        c = patches.Circle(xy=(self.center[0], self.center[1]), radius=self.radius, fc='r', ec='r')
        ax[2].add_patch(c)
        ax[2].plot([-1,-1,1,1,-1],[-1,1,1,-1,-1],c='black')
        ax[2].plot([-1,1],[-1,1], ':',c='black', alpha=0.7)
        im = ax[2].scatter(x[:,0],x[:,1],c=k_val)
        ax[2].arrow(x=other[0],y=other[1],dx=other[2],dy=other[3],
                         width=0.002,head_width=0.05,
                         head_length=0.02,
                         length_includes_head=True,color='black')
        fig.colorbar(im)
        k_val0 = np.array([calk([i/100-1,i/100-1],
                0.1*np.array([np.cos(0),np.sin(0)])) for i in range(200)])

        # Cross section
        for i in range(6):
            deg = i*30 
            rad = np.deg2rad(i*10) 
            k_val = np.array([calk([i/100-1,i/100-1],
                    0.1*np.array([np.cos(rad),np.sin(rad)])) for i in range(200)])
            if i:
                ax[0].plot([j/100-1 for j in range(200)],
                           k_val,
                           label='diff='+str(deg)+'[deg]')

                ax[1].plot([j/100-1 for j in range(200)],
                           k_val-k_val0,
                           label='diff='+str(deg)+'[deg]')
            else:
                ax[0].plot([j/100-1 for j in range(200)],
                           k_val,  ':',
                           label='diff(base)='+str(deg)+'[deg]',
                           c='black')
                ax[1].plot([j/100-1 for j in range(200)],
                           k_val-k_val0,  ':',
                           label='diff(base)='+str(deg)+'[deg]',
                           c='black')
        ax[0].set_xlabel('x=y=')
        ax[1].set_xlabel('x=y=')
        ax[0].set_ylabel('k(x,x*_angle)')
        ax[1].set_ylabel('k(x,x*_angle)-k(x,x*_0)')
        ax[0].arrow(x=other[1], y=1.0, dx=other[2],dy=other[3],
                         width=0.002,head_width=0.05,
                         head_length=0.02,
                         length_includes_head=True,color='black')
        ax[0].legend()
        ax[1].legend()


    def vis_kernel(self,other=np.array([0.9,0.9, 0.1, 0.0]),vec=np.array([0.1,0.])):
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as plt
        import mpl_toolkits.mplot3d.art3d as art3d
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as plt
        import mpl_toolkits.mplot3d.art3d as art3d
        import numpy as np
        import gpflow
        def func1(x, y,k,other):
            ar = []
            for xx, yy in zip(x,y):
                arr = []
                for xxx, yyy in zip(xx,yy):
                    arr.append(k.compute_K(np.array([[xxx,yyy, vec[0], vec[1]]]), 
                               np.zeros((1,4)) + other.reshape(1,4))[0][0])
                ar.append(arr)
            return np.array(ar)
                               
        k = gpflow.kernels.RBF(input_dim=4, variance=1., ARD=True)
        x = np.arange(-1.0, 1.0, 0.1)
        y = np.arange(-1.0, 1.0, 0.1)
        X, Y = np.meshgrid(x, y)
        Z = func1(X, Y,k,other)
        fig = plt.figure(figsize=(10,10))
        ax = Axes3D(fig)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("k(x, x*)")
        name = 'vec1:green('+str(other[2])+','+str(other[3])+') vs vec2:blue('+str(vec[0])+','+str(vec[1])+')'
        ax.set_title(name)
        ax.plot_surface(X, Y, Z)
        # ax.scatter([other[0]],[other[1]],[0.],marker = 'o',color='green',s=100)
        ax.plot([-1.,-1.,1.,1.,-1.],[-1.,1.,1.,-1.,-1.],[0.,0.,0.,0.,0.],c='black')
        c = patches.Circle(xy=(self.center[0], self.center[1]), radius=self.radius, fc='r', ec='r')
        ax.add_patch(c)
        ax.view_init(30, 40)
        art3d.pathpatch_2d_to_3d(c, z=0, zdir="z")
        ax.quiver(other[0], other[1], 0.0 , vec[0], vec[1], 0.0, color='green')
        ax.invert_xaxis()
        ax.set_zlim(-0.01,1.01)
        plt.show()

    def vis_trace(self,X,Y):
        fig = plt.figure(figsize=(5,5))
        ax = plt.axes()
        ax.plot([-1,-1,1,1,-1],[-1,1,1,-1,-1],c='black')
        c = patches.Circle(xy=(self.center[0], self.center[1]), radius=self.radius, fc='r', ec='r')
        ax.add_patch(c)
        for x,y in zip(X,Y):
            if  y[0] == 0 and y[1] == 0:
                ax.scatter(x[0],x[1],c='black',s=5)
            else:
                ax.arrow(x=x[0],y=x[1],dx=y[0],dy=y[1],
                         width=0.002,head_width=0.05,
                         head_length=0.02,
                         length_includes_head=True,color='k')
        ax.scatter(self.goal[0],self.goal[1],marker='x',c='black')

    def vis_scatter(self,X):
        fig = plt.figure(figsize=(5,5))
        ax = plt.axes()
        ax.plot([-1,-1,1,1,-1],[-1,1,1,-1,-1],c='black')
        c = patches.Circle(xy=(self.center[0], self.center[1]), radius=self.radius, fc='r', ec='r')
        ax.add_patch(c)
        ax.scatter(X[:,0],X[:,1],c='black',s=1,alpha=0.5)
        ax.scatter(self.goal[0],self.goal[1],marker='x',c='black')
