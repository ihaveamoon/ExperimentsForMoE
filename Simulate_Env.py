import gym
import numpy as np
from gym.utils import EzPickle
from uniform_instance import override
from Params import configs
from agent_utils import vanilla_placement
from copy import deepcopy
import torch
import random
import matplotlib.pyplot as plt
import time


def GenerateAFandTR(data,n_moe_layer,experts_per_layer):
    expert_aff =  np.zeros((experts_per_layer*n_moe_layer,experts_per_layer*n_moe_layer)) 
    expert_traf = np.zeros((experts_per_layer*n_moe_layer,experts_per_layer*n_moe_layer)) 
    for layer in range(n_moe_layer - 1):
        start_index = layer * experts_per_layer
        # end_index = start_index + experts_per_layer

        down_start_index = (layer + 1) * experts_per_layer
        # down_end_index = down_start_index + experts_per_layer

        random_weights = np.random.rand(experts_per_layer, experts_per_layer)
        row_sums = random_weights.sum(axis=1, keepdims=True)
        normalized_weights = random_weights / row_sums
        for i in range(experts_per_layer):
            for j in range(experts_per_layer):
                expert_aff[start_index + i][down_start_index + j] = normalized_weights[i, j]
                expert_traf[start_index + i][down_start_index + j] = data[start_index + i, down_start_index + j]
    
    return torch.tensor(expert_aff,dtype=torch.float32),torch.tensor(expert_traf,dtype=torch.float32)

def GeneratePair(n_e,n_g,v_p):
    history_expert_gpu = np.zeros((n_e, n_g), dtype=bool)
    for expert_id in range(n_e):
            for gpu_id in range(n_g):
                if v_p[expert_id][gpu_id]:
                    history_expert_gpu[expert_id][gpu_id] = True
    return history_expert_gpu

def GenerateGPtraffic(data,n_g,n_e,e_g):
    gpu_links = np.zeros((n_g, n_g))
    gpu_total_traffic = np.zeros((n_g,))
    for gpu_id in range(n_g):
            for other_gpu_id in range(n_g):
                if gpu_id != other_gpu_id:
                    for expert_id in range(n_e):
                        for other_expert_id in range(n_e):
                            if e_g[expert_id][gpu_id] and e_g[other_expert_id][other_gpu_id]:
                                traffic = data[expert_id, other_expert_id]
                                gpu_links[gpu_id, other_gpu_id] += traffic
                                gpu_total_traffic[gpu_id] += traffic
    return gpu_links,gpu_total_traffic

def GenerateGPULink(n_g,g_l,g_t):
    for gpu_id in range(n_g):
            if g_t[gpu_id] > 0:  # 避免除以0
                g_l[gpu_id, :] /= g_t[gpu_id]
    return torch.tensor(g_l, dtype=torch.float32,requires_grad=False)

class Simulate_Env(gym.Env, EzPickle):
    def __init__(self,
                 n_moe_layer,
                 n_e,
                 n_g):
        EzPickle.__init__(self)

        self.step_count = 0
        self.n_moe_layer = n_moe_layer
        self.number_of_experts = n_e
        self.number_of_gpus = n_g
        # the task id for first column
        self.candidate = []
        #self.expert_affinity = getExpertAffinity # 专家亲和力矩阵 待完成！！


    @override
    def reset(self, data,number_of_replicas = 4): 
        # 重置各类计数器和矩阵，为新的环境做准备
        # data (n_expert, n_expert)
        '''
        self.expert_traf:np array, the traffic between experts
        self.expert_aff:np array ,the affinity between experts
        self.gpu_link:np array ,the traffic between gpus
        self.current_token:np array,the token scattered to different experts
        有几个硬性的变量需要设计：
        1.mask操作，如何mask，如何计算显存的变化
        2.current_token是否和history_popularity重复了
        history_popularity指什么，若是指历史数据，应该在step的初始阶段
        '''

        experts_per_layer = self.number_of_experts // self.n_moe_layer

        self.step_count = 0
        # token tracked between experts
        self.expert_token = data.astype(np.single) 
        # rewards to be return after action was commited
        self.posRewards = 0
        # generate the pair of expert bonded to gpu
        self.vanilla_p,self.replica_p = vanilla_placement(self.number_of_experts, self.number_of_gpus,number_of_replicas) # (n_expert, n_gpu)
        # initialize the traffic between experts,the affinity between experts
        self.expert_traf,self.expert_aff = GenerateAFandTR(data,self.n_moe_layer,experts_per_layer)
        # initialize history_expert_gpu,history_gpu_expert according to vanilla_placement
        self.history_expert_gpu = GeneratePair(self.number_of_experts,self.number_of_gpus,self.vanilla_p)        
        # initialize self.gpu_links matrix : traffic rate
        self.gpu_links,self.gpu_total_traffic = GenerateGPtraffic(data,self.number_of_gpus,
                                                                  self.number_of_experts,self.history_expert_gpu)
        # 然后计算比例
        
        for gpu_id in range(self.number_of_gpus):
            if self.gpu_total_traffic[gpu_id] > 0:  # 避免除以0
                self.gpu_links[gpu_id, :] /= self.gpu_total_traffic[gpu_id]
        self.gpu_links = torch.tensor(self.gpu_links, dtype=torch.float32)

        self.gpu_links = GenerateGPULink(self.number_of_gpus,self.gpu_links,self.gpu_total_traffic)
        # 验证每个GPU的发送比例总和是否为1
        # for k in range(self.batch_size):
        #     for gpu_id in range(self.number_of_gpus):
        #         assert torch.isclose(torch.sum(self.gpu_links[k, gpu_id, :]), torch.tensor(1.0), atol=1e-6), f"Sum of proportions for GPU {gpu_id} in batch {k} is not 1."

        # current token routing
        self.current_token = np.zeros((self.number_of_experts,1))
        
        for i in range(self.number_of_experts): 
            total_tokens = np.sum(data[i, :]) - data[i, i]   # 专家i发送给所有其他专家的token总数
            self.current_token[i] = total_tokens         #总共的token数目

        # 初始化 expert_nodes: historical popularity、current token load
        self.history_popularity = np.random.uniform(low=0.1, high=1.0, size=(self.number_of_experts,1))
        
        self.expert_nodes = np.concatenate(
            [   self.current_token.reshape(self.number_of_experts, 1),
                self.history_popularity.reshape(self.number_of_experts, 1)],
            axis=-1  # 沿最后一个维度拼接 
        )
        self.expert_nodes = torch.tensor(self.expert_nodes, dtype=torch.float32)
        # print('reset() : expert_nodes[0]', self.expert_nodes[0])

        # 初始化 GPU nodes: compute speed、utilization、available memory
        compute_speed = np.random.uniform(low=60, high=80, size=(self.number_of_gpus,1)) # 假设计算速度在 60(TFLOPS) 到 80(TFLOPS)之间
        utilization = np.random.uniform(low=0.5, high=0.9, size=(self.number_of_gpus,1)) # 内存使用率
        total_memory = np.random.uniform(low=16, high=24, size=(self.number_of_gpus,1))  # 假设内存范围在 16GB 到 24GB 之间
        total_memory = np.zeros((self.number_of_gpus,1))
        total_memory.fill(24)                               #设置显存总容量为24GB
        # utilization = np.zeros((self.number_of_gpus,1)).fill(0.5)
        used_memory = total_memory * utilization
        available_memory = total_memory - used_memory

        self.gpu_nodes = np.concatenate(
            [compute_speed.reshape(self.number_of_gpus, 1), 
            utilization.reshape(self.number_of_gpus, 1), 
            available_memory.reshape(self.number_of_gpus, 1)], 
            axis=-1)
        self.gpu_nodes = torch.tensor(self.gpu_nodes, dtype=torch.float32)

        # initialize self.mask_expert, mask out current traffic < 200 / 1000
        self.mask_expert = self.current_token < 500
        # initialize self.mask_gpu, mask out utilization > 0.9
        self.mask_gpu = utilization > 0.9

        self.initQuality = 1

        return self.expert_traf, self.expert_nodes, self.expert_aff, self.gpu_links, self.gpu_nodes, self.mask_expert, self.mask_gpu


    def done(self):
        return np.all(self.mask_gpu)

    @override
    def step(self, expert_indices, gpu_bool_array, data, gantt_plt=None):
        '''
        G : GPU集合
        P : palce policy

        '''
        # 执行动作，维护和更新环境状态，计算奖励,注意,此时向环境提交了动作，
        # 应该先执行动作，再进行一个batch的训练，从而更新奖励，我们需要设计一个计算延迟的公式，还有一个计算switchcost的公式
        reward = 0
        #对于每轮输入的token数据以及expert,gpu之间的关系,按照策略进行调整，并且得到奖励
        expert_selected = expert_indices
        gpu_selected = gpu_bool_array.squeeze()
        print(gpu_selected)
        self.total_cost = 0
        self.current_token = data
        # update mask_expert :
        popularity_threshold = 0.1 # 假设我们根据 history_popularity 小于某个阈值来决定是否屏蔽
        # 在环境更新前,actor根据之前的环境已经生成了带mask的action，因此判断一个expert/gpu是不是被mask应该放到新环境之后再说

        #在gpu = True的位置上添加专家副本，同时如果对应的gpu为False,我们缩减掉该位置上的expert
        expand_cost = 1111
        shrink_cost = 0
        throughput = 10220
        #按照置空法实现replica增加和减少，data生成需要加噪，涵盖所有真实的变化
        print(self.mask_expert)
        print(self.mask_gpu)
                # update expert_nodes -> history popularity
        for expert in range(self.number_of_experts):
            self.history_popularity[expert] = self.history_popularity[expert] * 0.9 + self.current_token[expert] * 0.1

        if self.mask_expert[expert_selected] == True:
            for gpu in range(self.number_of_gpus):
                if self.mask_gpu[gpu] == True:
                    continue
                if gpu_selected[gpu] == True:
                    self.replica_p[expert_selected].append(gpu)
                    rewards += expand_cost
                elif gpu_selected[gpu] == False:
                    if gpu in self.replica_p[expert_selected]:
                        self.replica_p[expert_selected].remove(gpu)
                    reward += shrink_cost


        #根据最新的数据路由计算获取的奖励，已知expert的放置        
        for i in range(self.number_of_experts): 
            total_tokens = np.sum(data[i, :]) - data[i, i]   # 专家i发送给所有其他专家的token总数
            self.current_token = total_tokens         #总共的token数目

        previous_traffic = torch.sum(self.gpu_links)
        done = False
        #计算每个专家副本需要处理的token数目
        expert_cap = np.zeros((self.number_of_experts,))
        
        for expert in range(self.number_of_experts):
            tot_ep = len(self.replica_p[expert]) + 1
            expert_cap[expert] = self.current_token[expert] // tot_ep

        print(expert_cap)
        #动作已提交，计算环境的变化


        # redundant expert_gpu action makes no effect
        for gpu in range(self.number_of_gpus):
            for expert in range(self.number_of_experts):
                if self.history_expert_gpu[expert,gpu] == True:

           
            # update gpu_nodes : compute speed(stable), utilization, available memory
            old_utilization = deepcopy(self.gpu_nodes[gpu, 1])
            old_available_memory = deepcopy(self.gpu_nodes[gpu, 2])
            
            # if (old_utilization > 0.9) or (old_available_memory < 0): # 如果超过利用率或者内存不足，不更新
            #     break

            self.gpu_nodes[gpu, 1] = np.clip(old_utilization, 0, 1)  # 更新utilization并确保不超过100%
            self.gpu_nodes[gpu, 2] = np.clip(old_available_memory, 0, None)  # 更新available_memory并确保不为负

            # update gpu_links: bandwidth(stable), token traffic
            

            # update mask_gpu
            


        # update expert_nodes -> expert traffic
        alpha = 0.6  # 调节当前数据和历史数据的权重
        for ii in range(self.number_of_experts):
            total_tokens = np.sum(data[ii, :])
            if total_tokens > 0:
                for j in range(self.number_of_experts):
                    current_ratio = data[ii, j] / total_tokens
                    self.expert_traf[ii, j] = alpha * current_ratio + (1 - alpha) * self.expert_traf[ii, j]
            else:
                self.expert_traf[ii, :] = self.expert_traf[ii, :] * (1 - alpha)
        
        # update rewards : 如果跨GPU传输的数据量减少，则给予正奖励
        current_traffic = torch.sum(self.gpu_links)
        reward = 0
        if current_traffic < previous_traffic:
            reward = previous_traffic - current_traffic
        # print(reward)
        rewards.append(reward)

        self.expert_nodes = np.concatenate([
            self.current_token.reshape(self.number_of_experts, 1),
            self.history_popularity.reshape(self.number_of_experts, 1)
        ], axis=1)

        #环境的返回值
        return self.expert_nodes, self.expert_traf, self.gpu_nodes, self.gpu_links, self.mask_expert, self.mask_gpu, rewards