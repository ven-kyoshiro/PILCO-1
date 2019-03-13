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
        self.whereami = copy.deepcopy(self.ini_posi)
        self.goal = np.array([0.9,0.9])
        self.center = np.array([0.0,0.0])
        self.radius = radius
        self.timelimit =40
        self.N = 30 # Collision determination　resolution
        high = np.ones(2)*1
        self.observation_space = gym.spaces.Box(low=-np.ones(2)*1, high=np.ones(2)*1,dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-np.ones(2)*0.1, high=np.ones(2)*0.1,dtype=np.float32)

    def reset(self):
        self.timestep = 0
        self.whereami = copy.deepcopy(self.ini_posi)+np.random.normal(0.0, 0.005, 2)
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
        norm = np.linalg.norm(self.whereami-self.goal)
        rew = np.exp(-(norm**2)*1)
        return self.whereami, rew, self.timestep>=self.timelimit,{}

    def render(self):
        fig = plt.figure(figsize=(5,5))
        ax = plt.axes()
        ax.plot([-1,-1,1,1,-1],[-1,1,1,-1,-1],c='black')
        c = patches.Circle(xy=(self.center[0], self.center[1]), radius=self.radius, fc='r', ec='r')
        ax.add_patch(c)
        ax.scatter(self.whereami[0],self.whereami[1],c='black')
        ax.scatter(self.goal[0],self.goal[1],marker='x',c='black')

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
