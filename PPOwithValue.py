from agent_utils import eval_actions
from agent_utils import select_gpus
from models.PPO_Actor2 import Expert_Encoder, GPU_Encoder, Expert_Decoder, GPU_Decoder, Expert_Actor, GPU_Actor, MLPCritic
from copy import deepcopy
import torch
import time
from torch.distributions.categorical import Categorical
import torch.nn as nn
import numpy as np
from Params import configs
from validation import validate
import os
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Memory:
    # 待修改！
    def __init__(self):
        self.expert_node_fea = [] # expert 特征矩阵
        self.expert_link_fea = [] # expert 邻接矩阵
        self.expert_selection = [] # 上一次调度的 expert indice
        self.mask_expert = []
        self.expert_logprobs = [] # expert 的 log 概率  

        self.gpu_node_fea = [] # gpu 特征矩阵
        self.gpu_link_fea = [] # gpu 邻接矩阵
        self.gpu_selection = []
        self.mask_gpu = []
        self.gpu_logprobs = [] # gpu 的 log 概率
        
        self.env_rewards = [] # 奖励
        self.env_done = []


    def clear_memory(self):
        del self.expert_link_fea[:]
        del self.expert_node_fea[:]
        del self.expert_selection[:]
        del self.mask_expert[:]
        del self.expert_logprobs[:]

        del self.gpu_node_fea[:]
        del self.gpu_link_fea[:]
        del self.gpu_selection[:]
        del self.mask_gpu[:]
        del self.gpu_logprobs[:]
        
        del self.env_rewards[:]
        del self.env_done[:]


class PPO:
    def __init__(self,
                 lr,
                 gamma,
                 k_epochs,
                 eps_clip,
                 n_moe_layer, # number of moe layers
                 n_e, # the total number of experts 
                 n_g, # number of gpus
                 num_layers, # for GCNN, number of layers in the neural networks (INCLUDING the input layer), 每一层可能使用一个 MLP 来处理节点的特征
                 neighbor_pooling_type, 
                 input_dim,
                 hidden_dim,
                 num_mlp_layers_feature_extract, # for GCNN, number of layers in mlps (EXCLUDING the input layer), 指定了每个 MLP 的内部层数
                 num_mlp_layers_actor,
                 hidden_dim_actor,
                 num_mlp_layers_critic,
                 hidden_dim_critic,
                 ):
        self.lr = lr
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.expert_encoder = Expert_Encoder(
                                    expert_feature_dim = configs.expert_feature_dim,
                                    hidden_dim = configs.hidden_dim,
                                    output_dim=configs.expert_output_dim,
                                    num_layers = configs.num_layers, 
                                    num_mlp_layers = configs.num_mlp_layers_feature_extract).to(device)
        self.gpu_encoder = GPU_Encoder(
                                    gpu_feature_dim = configs.gpu_feature_dim,
                                    hidden_dim = configs.hidden_dim, 
                                    output_dim=configs.gpu_output_dim,
                                    num_layers = configs.num_layers,
                                    num_mlp_layers=configs.num_mlp_layers_feature_extract).to(device)
        self.expert_decoder = Expert_Decoder(
                                    input_dim = configs.expert_output_dim + configs.gpu_output_dim + configs.expert_output_dim,
                                    hidden_dim = configs.hidden_dim,
                                    output_dim = 1,
                                    num_layers = configs.num_mlp_layers_actor).to(device)
        self.gpu_decoder = GPU_Decoder(
                                    input_dim = configs.expert_output_dim + configs.gpu_output_dim + configs.gpu_output_dim,
                                    hidden_dim = configs.hidden_dim,
                                    output_dim = 1,
                                    num_layers = configs.num_mlp_layers_actor).to(device)
        self.expert_actor = Expert_Actor(self.expert_encoder,self.expert_decoder).to(device)
        self.gpu_actor = GPU_Actor(self.gpu_encoder,self.gpu_decoder).to(device)

        self.policy_critic = MLPCritic(num_layers = num_mlp_layers_critic, 
                                        input_dim = configs.output_dim + configs.n_g, # expert + gpu array
                                        hidden_dim = configs.hidden_dim, 
                                        output_dim = 1).to(device)
        self.policy_old_expert = deepcopy(self.expert_actor)
        self.policy_old_gpu = deepcopy(self.gpu_actor)
    
        self.policy_old_expert.load_state_dict(self.expert_actor.state_dict())
        self.policy_old_gpu.load_state_dict(self.gpu_actor.state_dict())

        self.expert_optimizer = torch.optim.Adam(self.expert_actor.parameters(), lr=lr)
        self.gpu_optimizer = torch.optim.Adam(self.gpu_actor.parameters(), lr=lr)
        self.value_optimizer = torch.optim.Adam(self.policy_critic.parameters(), lr=lr)

        self.expert_scheduler = torch.optim.lr_scheduler.StepLR(self.expert_optimizer, step_size=configs.decay_step_size, gamma=configs.decay_ratio)
        self.gpu_scheduler = torch.optim.lr_scheduler.StepLR(self.gpu_optimizer, step_size=configs.decay_step_size, gamma=configs.decay_ratio)
        self.value_scheduler = torch.optim.lr_scheduler.StepLR(self.value_optimizer, step_size=configs.decay_step_size, gamma=configs.decay_ratio)

        self.MSE = nn.MSELoss()


    def update(self, memories, epoch):
        '''self.policy_expert.train()
        self.policy_gpu.train()
        self.policy_critic.train()'''

        vloss_coef = configs.vloss_coef
        ploss_coef = configs.ploss_coef
        entloss_coef = configs.entloss_coef
        rewards_all_env = []

        # 计算折扣奖励并进行标准化
        for rewards_list, dones_list in zip(memories.env_rewards, memories.env_done):
            rewards = []
            discounted_reward = 0

            rewards_list = rewards_list.squeeze() # 转换为一维张量
            dones_list = dones_list.squeeze()

            for reward, is_terminal in zip(reversed(rewards_list), reversed(dones_list)):
                if is_terminal:
                    discounted_reward = 0
                discounted_reward = reward + (self.gamma * discounted_reward)
                rewards.insert(0, discounted_reward)

            rewards = torch.tensor(rewards, dtype=torch.float).to(device)
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
            rewards_all_env.append(rewards)

        rewards_all_env = torch.stack(rewards_all_env, 0).squeeze()
        
        for _ in range(configs.k_epochs):
            loss_sum = 0
            vloss_sum = 0
            
            expert_log_prob = []
            gpu_log_prob = []
            val = []
            gpu_select = []
            entropies = []
            expert_entropy = []
            gpu_entropy = []
            
            expert_log_old_prob = memories.expert_logprobs[0]
            gpu_log_old_prob = memories.gpu_logprobs[0]

            for i in range(len(memories.expert_node_fea)):
                env_expert_nodes = memories.expert_node_fea[i]
                env_expert_links = memories.expert_link_fea[i]
                env_gpu_nodes = memories.gpu_node_fea[i]
                env_gpu_links = memories.gpu_link_fea[i]
                env_mask_expert = memories.mask_expert[i]
                env_mask_gpu = memories.mask_gpu[i]

                expert_prob,expert_indices = self.policy_expert(
                                                        node_features = env_expert_nodes,
                                                        adj_matrix = env_expert_links,
                                                        mask= env_mask_expert)
                # selected_expert_features = env_expert_nodes[:, expert_indices[0], :] # torch.Size([64, 2])
                # selected_expert_links = env_expert_links[:, expert_indices[0], :] # torch.Size([64, 32])

                gpu_prob, gpu_bool_array = self.policy_gpu(
                                                gpu_nodes = env_gpu_nodes, 
                                                gpu_links = env_gpu_links, 
                                                # pooling_type = configs.graph_pool_type,
                                                mask_gpu_action = env_mask_gpu)
                print("\nexpert_prob[batch 0] = ", expert_prob[0], "\ngpu_prob[batch 0] = ", gpu_prob[0], "\n")
                # Combine (action_e, action_g)
                critic_input = torch.cat([h_pooled, gpu_bool_array.float()], dim=1)
                v = self.policy_critic(critic_input)
                val.append(v)

                # Calculate the log probabilities
                expert_log_prob.append(torch.log(expert_prob + 1e-10))
                gpu_log_prob.append(torch.log(gpu_prob + 1e-10))

                # Calculate the entropies
                expert_dist = Categorical(expert_prob)
                expert_entropy.append(expert_dist.entropy())

                gpu_dist = Categorical(gpu_prob)
                gpu_entropy.append(gpu_dist.entropy())

            # Convert lists to tensors
            expert_log_prob, expert_log_old_prob = torch.cat(expert_log_prob, dim=0), torch.cat(expert_log_old_prob, dim=0) # torch.Size([64, 32])
            gpu_log_prob, gpu_log_old_prob = torch.cat(gpu_log_prob, dim=0), torch.cat(gpu_log_old_prob, dim=0) # torch.Size([64, 4])
            
            val = torch.cat(val).squeeze() # torch.Size([64])
            expert_entropy = torch.cat(expert_entropy).squeeze() # torch.Size([64])
            gpu_entropy = torch.cat(gpu_entropy).squeeze() # torch.Size([64])

            # Compute advantages
            advantages = rewards_all_env - val.detach()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Policy loss
            expert_loss_sum = torch.zeros(1, device=device)
            gpu_loss_sum = torch.zeros(1, device=device)
            value_loss_sum = torch.zeros(1, device=device)
            for j in range(configs.batch_size):
                expert_ratios = torch.exp(expert_log_prob[j] - expert_log_old_prob[j].detach()) # torch.Size([32])
                gpu_ratios = torch.exp(gpu_log_prob[j] - gpu_log_old_prob[j].detach()) #  torch.Size([4])

                expert_surr1 = expert_ratios * advantages[j] # torch.Size([32])
                expert_surr2 = torch.clamp(expert_ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages[j] # torch.Size([32])
                expert_loss = -1 * torch.min(expert_surr1, expert_surr2) - entloss_coef * expert_entropy[j]
                expert_loss_sum += expert_loss.sum() # torch.Size([1])

                gpu_surr1 = gpu_ratios * advantages[j]
                gpu_surr2 = torch.clamp(gpu_ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages[j]
                gpu_loss = -1 * torch.min(gpu_surr1, gpu_surr2) - entloss_coef * gpu_entropy[j]
                gpu_loss_sum += gpu_loss.sum() # torch.Size([1])

                value_loss = self.MSE(val[j], rewards_all_env[j])
                value_loss_sum += value_loss # torch.Size([1])

            # Calculate the total loss
            total_expert_loss = ploss_coef * expert_loss_sum / configs.batch_size
            total_gpu_loss = ploss_coef * gpu_loss_sum / configs.batch_size
            total_value_loss = vloss_coef * value_loss_sum / configs.batch_size

            # take gradient step, scheduler.step()
            self.expert_optimizer.zero_grad()
            total_expert_loss.backward(retain_graph=True)
            self.expert_optimizer.step()

            self.gpu_optimizer.zero_grad()
            total_gpu_loss.backward(retain_graph=True)
            self.gpu_optimizer.step()

            self.value_optimizer.zero_grad()
            total_value_loss.backward()
            self.value_optimizer.step()

            # Copy new weights into old policy
            self.policy_old_expert.load_state_dict(self.policy_expert.state_dict())
            self.policy_old_gpu.load_state_dict(self.policy_gpu.state_dict())
            
            if configs.decayflag:
                self.expert_scheduler.step()
                self.gpu_scheduler.step()
                self.value_scheduler.step()

            return expert_loss_sum.mean().item(), gpu_loss_sum.mean().item(), value_loss_sum.mean().item()
    

def main(epochs):
    from uniform_instance import Simulate_Dataset
    from Simulate_Env import Simulate_Env

    log = []
    # 初始化PPO算法的参数, 并配置与 动态调度 问题相关的参数
    ppo = PPO(configs.lr, configs.gamma, configs.k_epochs, configs.eps_clip,
              n_moe_layer = configs.n_moe_layer,
              n_e=configs.n_e,
              n_g=configs.n_g,
              num_layers=configs.num_layers,
              neighbor_pooling_type=configs.neighbor_pooling_type,
              input_dim=configs.input_dim,
              hidden_dim=configs.hidden_dim,
              num_mlp_layers_feature_extract=configs.num_mlp_layers_feature_extract,
              num_mlp_layers_actor=configs.num_mlp_layers_actor,
              hidden_dim_actor=configs.hidden_dim_actor,
              num_mlp_layers_critic=configs.num_mlp_layers_critic,
              hidden_dim_critic=configs.hidden_dim_critic)
    # 这里是随机生成的样本，需修改模拟器！
    simu_tokens = 200 # 假设每对专家之间最多有200个token
    n_e_per_layer = configs.n_e / configs.n_moe_layer
    train_dataset = Simulate_Dataset(n_e_per_layer, configs.n_moe_layer, simu_tokens, configs.num_ins)
    validat_dataset = Simulate_Dataset(n_e_per_layer, configs.n_moe_layer, simu_tokens, 64)
    
    # [样本数, 专家数, 专家数]: 样本k中，专家i到专家j需要路由的token数量
    data_loader = DataLoader(train_dataset, batch_size=configs.batch_size)
    valid_loader = DataLoader(validat_dataset, batch_size=configs.batch_size)

    record = 1000000
    for epoch in range(epochs):
        memory = Memory() # 存储过程数据
        ppo.policy_old_expert.train()
        ppo.policy_old_gpu.train()

        times, losses, rewards2, critic_rewards = [], [], [], []
        start = time.time()

        costs = []
        expert_losses, gpu_losses, rewards, critic_loss = [], [], [], []
        for batch_idx, batch in enumerate(data_loader):
            env = Simulate_Env(configs.n_moe_layer, configs.n_e, configs.n_g)
            data = batch.numpy() # torch.Size([64, 32, 32])

            # env.reset函数
            expert_links, expert_nodes, expert_adj, gpu_links, gpu_nodes, mask_expert, mask_gpu = env.reset(data)

            expert_log_prob = []
            gpu_log_prob = []
            env_rewards = []
            env_done = [] # 若所有GPU负载已满，则结束动态调度?

            gpu_select = []
            pool = None
            ep_rewards = - env.initQuality
            
            env_mask_expert = torch.from_numpy(np.copy(mask_expert)).to(device)
            env_mask_gpu = torch.from_numpy(np.copy(mask_gpu)).to(device)
            while True:
                env_expert_links = deepcopy(torch.Tensor(expert_links)).to(device) # torch.Size([batch siez, n_e, n_e])
                env_expert_nodes = deepcopy(torch.Tensor(expert_nodes)).to(device) # torch.Size([batch siez, n_e, fea_dim = 2])
                env_expert_adj = deepcopy(expert_adj).to(device) # torch.Size([batch siez, n_e, n_e])

                env_gpu_links = deepcopy(gpu_links).to(device) # torch.Size([batch siez, n_g, n_g])
                env_gpu_nodes = deepcopy(gpu_nodes).to(device) # torch.Size([batch siez, n_g, fea_dim = 3])
                # print("env_gpu_links = ", env_gpu_links[0][0], "\n")
                # print("env_gpu_nodes = ", env_gpu_nodes[0][0], "\n")

                # Encode expert and GPU states
                h_expert, h_pooled_expert = ppo.expert_encoder(env_expert_nodes, env_expert_adj)
                h_gpu, h_pooled_gpu = ppo.gpu_encoder(env_gpu_nodes, env_gpu_links)
                # print("h_expert = ", h_expert[0], "\n")
                # print("h_gpu = ", h_gpu[0], "\n")
                # print("h_pooled_expert = ", h_pooled_expert[0], "\n")
                # print("h_pooled_gpu = ", h_pooled_gpu[0], "\n")
                # Get action decisions from actors
                expert_action_probs, selected_expert_id = ppo.expert_actor(env_expert_nodes, env_expert_adj,env_mask_expert)
                selected_expert_embeddings = h_expert[torch.arange(h_expert.size(0)), selected_expert_id]

                gpu_action_probs, gpu_bool_array = ppo.gpu_actor(env_gpu_nodes, env_gpu_links,env_mask_gpu)
                # print("expert_action_probs = ", expert_action_probs, "\ngpu_action_probs = ", gpu_action_probs, "\n")
                # print("selected_expert_id = ", selected_expert_id, "\ngpu_bool_array = ", gpu_bool_array, "\n")
                print(selected_expert_id.shape)
                print(gpu_bool_array.shape)
                # 记录过程数据
                memory.expert_selection.append(selected_expert_id)
                memory.gpu_selection.append(gpu_bool_array)

                memory.expert_node_fea.append(env_expert_nodes)
                memory.expert_link_fea.append(env_expert_links)
                expert_log_prob.append(expert_action_probs)
                
                memory.gpu_node_fea.append(gpu_nodes)
                memory.gpu_link_fea.append(gpu_links)
                gpu_log_prob.append(gpu_action_probs)
                
                # 向环境提交选择的动作和机器，接收新的状态、奖励和完成标志等信息, 待修改！！！
                expert_nodes, expert_links, gpu_nodes, gpu_links, mask_expert, mask_gpu, dur_time, gpu_done, reward = env.step(selected_expert_id,
                                                                                                gpu_bool_array,
                                                                                                data)
                ep_rewards += reward

                env_rewards.append(deepcopy(reward))
                env_done.append(deepcopy(gpu_done))
                print('env step() : dur_time', dur_time)
                if env.done(): # mask_gpu 没有可用的GPU时，结束
                    break
            
            memory.mask_expert.append(env_mask_expert)
            memory.mask_gpu.append(env_mask_gpu)

            memory.expert_logprobs.append(expert_log_prob)
            memory.gpu_logprobs.append(gpu_log_prob)

            print("memory.env_done = ", env_done, "\n")
            memory.env_rewards.append(torch.tensor(env_rewards).float().permute(1, 0))
            memory.env_done.append(torch.tensor(env_done).float().permute(1, 0))

            # rewards
            ep_rewards -= env.posRewards
            # ppo.update
            torch.autograd.set_detect_anomaly(True)
            expert_loss, gpu_loss, value_loss = ppo.update(memory, batch_idx)

            memory.clear_memory()
            mean_time = np.mean(ep_rewards)
            log.append([batch_idx, mean_time])

            # 定期日志记录
            if batch_idx % 100 == 0:
                file_writing_obj = open(
                    './' + 'log_' + str(configs.n_e) + '_' + str(configs.n_g) + '_' + str(configs.low) + '_' + str(
                        configs.high) + '.txt', 'w')
                file_writing_obj.write(str(log))

            rewards.append(np.mean(ep_rewards).item())
            expert_losses.append(expert_loss)
            gpu_losses.append(gpu_loss)
            critic_loss.append(value_loss)

            cost = time.time() - start
            costs.append(cost)
            step = 20

            filepath = 'saved_network'
            # 定期模型保存
            if (batch_idx + 1) % step  == 0 :
                end = time.time()
                times.append(end - start)
                start = end
                expert_mean_loss = np.mean(expert_losses[-step:])
                gpu_mean_loss = np.mean(gpu_losses[-step:])
                mean_time = np.mean(costs[-step:])
                critic_losss = np.mean(critic_loss[-step:])

                filename = 'MultiPPO_{}'.format('Experts'+str(configs.n_e)+'GPU'+str(configs.n_g))
                filepath = os.path.join(filepath, filename)
                epoch_dir = os.path.join(filepath, '%s_%s' % (100, batch_idx))
                if not os.path.exists(epoch_dir):
                    os.makedirs(epoch_dir)
                expert_savePath = os.path.join(epoch_dir, '{}.pth'.format('policy_expert'))
                gpu_savePath = os.path.join(epoch_dir, '{}.pth'.format('policy_gpu'))

                torch.save(ppo.policy_expert.state_dict(), expert_savePath)
                torch.save(ppo.policy_gpu.state_dict(), gpu_savePath)

                print('  Batch %d/%d, mean_time: %2.3f, expert_loss: %2.4f, gpu_loss: %2.4f,critic_loss:%2.4f,took: %2.4fs' %
                      (batch_idx, len(data_loader), mean_time, expert_mean_loss, gpu_mean_loss, critic_losss,
                       times[-1]))

                # 性能评估与验证，用于实时监控模型的学习进度和性能
                t4 = time.time()
                validation_log = validate(valid_loader, configs.batch_size, ppo.policy_expert, ppo.policy_gpu).mean()
                if validation_log<record: # 保存最佳模型
                    epoch_dir = os.path.join(filepath, 'best_value100')
                    if not os.path.exists(epoch_dir):
                        os.makedirs(epoch_dir)
                    expert_savePath = os.path.join(epoch_dir, '{}.pth'.format('policy_expert'))
                    gpu_savePath = os.path.join(epoch_dir, '{}.pth'.format('policy_gpu'))
                    torch.save(ppo.policy_expert.state_dict(), expert_savePath)
                    torch.save(ppo.policy_gpu.state_dict(), gpu_savePath)
                    record = validation_log

                print('The validation quality is:', validation_log)
                file_writing_obj1 = open(
                    './' + 'vali_' + str(configs.n_e) + '_' + str(configs.n_g) + '_' + str(configs.low) + '_' + str(
                        configs.high) + '.txt', 'w')
                file_writing_obj1.write(str(validation_log))
                t5 = time.time()
        np.savetxt('./N_%s_M%s_u100'%(configs.n_e,configs.n_g),costs,delimiter="\n")


if __name__ == '__main__':
    total1 = time.time()
    main(1)
    total2 = time.time()

    #print(total2 - total1)
