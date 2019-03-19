import numpy as np
from pilco.models import PILCO 
from pilco.envs import JapanMaze 
from pilco.controllers import RbfController, LinearController
from pilco.rewards import ExponentialReward
from pilco.rewards import ExponentialReward
import tensorflow as tf
from tensorflow import logging
import os
import pickle
import matplotlib.pyplot as plt


np.random.seed(45)
expname='test5'


if not os.path.exists(expname):
    os.mkdir(expname)

def rollout(env,policy, timesteps):
    X = []; Y = []
    x = env.reset()
    for timestep in range(timesteps):
        u = policy(x)
        x_new, _, done, _ = env.step(u)
        if done: break
        X.append(np.hstack((x, u)))
        Y.append(x_new - x)
        x = x_new
    return np.stack(X), np.stack(Y)

def random_policy(x):
    return env.action_space.sample()

def pilco_policy(x):
    return pilco.compute_action(x[None, :])[0, :]

def remove_same_datum(X,Y):
    XY= np.hstack((X,Y))
    XY = np.unique(XY,axis=0)
    return XY[:,:X.shape[1]],XY[:,X.shape[1]:]

def vis_rews(rew_record,pred_record,expname):
    plt.figure()
    plt.plot(range(1,len(rew_record)+1),rew_record,label='real') 
    plt.plot(range(1,len(pred_record)+1),pred_record,label='predicted with GP') 
    plt.xlabel("after ** itr")
    plt.ylabel("sum of rewards/episode")
    plt.legend()
    plt.savefig(expname+"/sum_rews.png",format = 'png', dpi=100)
    with open(expname+'/sum_rews.pickle', mode='wb') as f:
        pickle.dump(rew_record, f)
    with open(expname+'/pred_sum_rews.pickle', mode='wb') as f:
        pickle.dump(pred_record, f)
    plt.close()

try:
    with tf.Session(graph=tf.Graph()) as sess:
        env = JapanMaze()  
        # Initial random rollouts to generate a dataset
        X,Y = rollout(env,policy=random_policy, timesteps=40)
        for i in range(1,3):
            X_, Y_ = rollout(env,policy=random_policy, timesteps=40)
            X = np.vstack((X, X_))
            Y = np.vstack((Y, Y_))
        state_dim = Y.shape[1]
        control_dim = X.shape[1] - state_dim
        controller = RbfController(state_dim=state_dim,
                                   control_dim=control_dim,
                                   num_basis_functions=4)
        R = ExponentialReward(state_dim=state_dim, t=np.array(env.goal))
        pilco = PILCO(X, Y, 
                      controller=controller,
                      horizon=40,
                      reward=R,
                      m_init = env.ini_posi.reshape(1,2), 
                      S_init = env.ini_cov)
        rew_record = []
        pred_record = []
        for r in range(40):
            itrname = str(r)
            # visualizations
            env.vis_trace(X,Y,save_name=expname + '/' + 'collected_trajectry_after_itr:'+itrname+'.png')
            env.vis_policy(pilco,save_name=expname + '/' + 'policy_behavior_after_itr:'+itrname+'.png')
            env.vis_gpr(pilco,save_name=expname + '/' + 'GPmodel_prediction_after_itr:'+itrname+'.png')
            if r:
                pred_record.append(pred_rew)
                rew_record.append(env.calc_sum_rews(X_new))
                vis_rews(rew_record,pred_record,expname)
                env.vis_trace(X_new,Y_new,save_name=expname + '/' + 'collected_trajectry_in_itr:'+itrname+'.png')
            # error handling
            for i in range(100):
                try:
                    pilco.optimize_models(restarts=1)
                    break
                except Exception as e:
                     print('model error trial:'+str(i)+'in roll:'+str(r))
                if i==99:
                    print(str(traceback.format_exc())+'\n')
                    raise
            for i in range(100):
                try:
                    pred_rew = pilco.optimize_policy(restarts=1,return_rews=True)
                    break
                except:
                    print('policy  error trial:'+str(i)+'in roll:'+str(r))
                if i==99:
                    raise

            X_new, Y_new = rollout(env,policy=pilco_policy, timesteps=40)
            # Update dataset
            X = np.vstack((X, X_new)); Y = np.vstack((Y, Y_new))
            X,Y = remove_same_datum(X,Y)
            pilco.mgpr.set_XY(X, Y)
        itrname = str(r+1)
        rew_record.append(env.calc_sum_rews(X_new))
        pred_record.append(pred_rew)
        vis_rews(rew_record,pred_record,expname)
        env.vis_trace(X,Y,save_name=expname + '/' + 'collected_trajectry_after_itr:'+itrname+'.png')
        env.vis_policy(pilco,save_name=expname + '/' + 'policy_behavior_after_itr:'+itrname+'.png')
        env.vis_gpr(pilco,save_name=expname + '/' + 'GPmodel_prediction_after_itr:'+itrname+'.png')
        env.vis_trace(X_new,Y_new,save_name=expname + '/' + 'collected_trajectry_in_itr:'+itrname+'.png')

except Exception as e:
    print(e)
    print(traceback.format_exc())
    print('!!!! Error !!!!\n'+str(e)+'\n')
    print(str(traceback.format_exc())+'\n')

