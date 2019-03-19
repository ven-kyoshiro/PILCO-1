import numpy as np
from pilco.envs import JapanMaze 
import tensorflow as tf
from tensorflow import logging
import os
import copy
import pickle
import matplotlib.pyplot as plt

import torch

seed = 41
np.random.seed(seed)
log_dir_name='ppo2'

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

### machina
from simple_net import PolNet,VNet
from machina.pols import CategoricalPol,GaussianPol
from machina.vfuncs import DeterministicSVfunc
from machina.traj import epi_functional as ef
from machina import logger
from machina.utils import measure
from machina.traj import Traj
from machina.algos import ppo_clip
from machina.samplers import EpiSampler

# define env
env = JapanMaze()
env_name = 'JapanMaze'
obs = env.reset()
ob_space = env.observation_space
ac_space = env.action_space
print('obs:', ob_space)
print('act:', ac_space)

# define policy
pol_net = PolNet(ob_space, ac_space)
pol = GaussianPol(ob_space, ac_space, pol_net)
vf_net = VNet(ob_space)
vf = DeterministicSVfunc(ob_space, vf_net)
pol_lr = 1e-4
optim_pol = torch.optim.Adam(pol_net.parameters(), pol_lr)
vf_lr = 3e-4
optim_vf = torch.optim.Adam(vf_net.parameters(), vf_lr)

# arguments of PPO
kl_beta = 1
gamma = 1.0# 0.995
lam = 1 
clip_param = 0.2
epoch_per_iter = 50
batch_size = 64
max_grad_norm = 10

# records
if not os.path.exists(log_dir_name):
    os.mkdir(log_dir_name)
score_file = os.path.join(log_dir_name, 'progress.csv')
logger.add_tabular_output(score_file)

# counter and record for loop
total_epi = 0
total_step = 0
max_rew = -500

# how long will you train
max_episodes = 3.2e4

sampler = EpiSampler(env, pol, num_parallel=4, seed=seed)

# main loop
saved_list=[]
obss = np.array([[]]) 
while max_episodes > total_epi:
    # sample trajectories
    with measure('sample'):
        epis = sampler.sample(pol, max_episodes=40)
    for ep in epis:
        if not obss.shape[1] == 2:
            obss = copy.deepcopy(ep['obs'])
        else:
            obss = np.vstack((obss,ep['obs']))
    # visualize per 1e2
    if not int(total_epi/1e2) in saved_list:
        saved_list.append(int(total_epi/1e2))
        env.vis_policy(pol,ppo=True,
                       save_name=log_dir_name + '/' + 'policy_behavior_after_itr:'+str(total_epi)+'.png')
        env.vis_trace(epis[0]['obs'][:-1],epis[0]['obs'][1:]-epis[0]['obs'][:-1],
                      save_name=log_dir_name + '/' + 'collected_trajectry_in_itr:'+str(total_epi)+'.png')
        if not -int(total_epi/1e3) in saved_list:
            if total_epi:
                saved_list.append(-int(total_epi/1e3))
                env.vis_scatter(obss,
                                save_name=log_dir_name + '/' + 'collected_trajectry_-1_to_itr:'+str(total_epi)+'.png')
                obss = np.array([[]]) 

    # train from trajectories
    with measure('train'):
        traj = Traj()
        traj.add_epis(epis)
        
        # calulate advantage
        traj = ef.compute_vs(traj, vf)
        traj = ef.compute_rets(traj, gamma)
        traj = ef.compute_advs(traj, gamma, lam)
        traj = ef.centerize_advs(traj)
        traj = ef.compute_h_masks(traj)
        traj.register_epis()

        result_dict = ppo_clip.train(traj=traj, pol=pol, vf=vf, clip_param=clip_param,
                                     optim_pol=optim_pol, optim_vf=optim_vf, 
                                     epoch=epoch_per_iter, batch_size=batch_size,
                                     max_grad_norm=max_grad_norm)
    # update counter and record
    total_epi += traj.num_epi
    step = traj.num_step
    total_step += step
    rewards = [np.sum(epi['rews']) for epi in epis]
    mean_rew = np.mean(rewards)
    logger.record_results(log_dir_name, result_dict, score_file,
                          total_epi, step, total_step,
                          rewards,
                          plot_title=env_name)

    del traj
del sampler
