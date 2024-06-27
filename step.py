def step(self, expert_indices, gpu_bool_array, data, gantt_plt=None):
        # 执行动作，维护和更新环境状态，计算奖励,注意,此时向环境提交了动作，
        # 应该先执行动作，再进行一个batch的训练，从而更新奖励，我们需要设计一个计算延迟的公式，还有一个计算switchcost的公式
        t1 = time.time()
        rewards, gpu_done = [],[]
        #对于每轮输入的token数据以及expert,gpu之间的关系,按照策略进行调整，并且得到奖励
        expert_selected = expert_indices
        gpu_selected = gpu_bool_array
        self.current_token = data
        # update mask_expert :
        popularity_threshold = 0.1 # 假设我们根据 history_popularity 小于某个阈值来决定是否屏蔽
        # 执行动作
        # 屏蔽expert,gpu不应被屏蔽
        for expert in range(self.number_of_experts):
            if self.history_popularity[expert] < popularity_threshold:
                self.mask_expert[expert] = True
            else:
                self.mask_expert[expert] = False
        if self.mask_expert[expert_selected] != True:
            for gpu in range(self.number_of_gpu):
                if gpu_selected[gpu] == True:
                    self.replica_p[expert_selected]._append() 

        previous_traffic = torch.sum(self.gpu_links)
        done = False
        '''若一个专家有多个 expert replica，还需要根据 token routing split 进行更新，待实现'''
        # redundant expert_gpu action makes no effect
        for j in range(self.number_of_gpus):
            
            # 检查 gpu_selected 是否被屏蔽，如果是则跳过本次循环
            if self.mask_gpu[j]:
                continue
            # UPDATE BASIC INFO 
            previous_placement = self.history_expert_gpu[expert_selected, j] # the chosen expert on gpu j history before update 
            replica_placement = self.replica_p[expert_selected]
            # update gpu_nodes : compute speed(stable), utilization, available memory
            old_utilization = deepcopy(self.gpu_nodes[j, 1])
            old_available_memory = deepcopy(self.gpu_nodes[j, 2])
            
            token_load = self.current_token[expert_selected]
            
            
            if (old_utilization > 0.9) or (old_available_memory < 0): # 如果超过利用率或者内存不足，不更新
                break
            self.history_expert_gpu[expert_selected, j] = gpu_selected[j] # update expert_gpu！！！

            self.gpu_nodes[j, 1] = np.clip(old_utilization, 0, 1)  # 更新utilization并确保不超过100%
            self.gpu_nodes[j, 2] = np.clip(old_available_memory, 0, None)  # 更新available_memory并确保不为负

            # update gpu_links: bandwidth(stable), token traffic
            for k in range(self.number_of_gpus):
                if k != j:
                    if previous_placement == 1:
                        self.gpu_links[j, k] -= self.current_token[expert_selected]
                        self.gpu_links[k, j] -= self.current_token[expert_selected]
                    if gpu_selected[j] == 1:
                        self.gpu_links[j, k] += self.current_token[expert_selected]
                        self.gpu_links[k, j] += self.current_token[expert_selected]
            
            # update mask_gpu
            self.mask_gpu[j] = self.gpu_nodes[j, 1] > 0.9
            done = self.done()
            
        # update expert_nodes : current token load(already updated), history popularity
        for expert in range(self.number_of_experts):
            self.history_popularity[expert] = self.history_popularity[expert] * 0.9 + self.current_token[expert] * 0.1

        # update expert_links :
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
        gpu_done.append(done)

        self.expert_nodes = np.concatenate([
            self.current_token.reshape(self.number_of_experts, 1),
            self.history_popularity.reshape(self.number_of_experts, 1)
        ], axis=1)

        t2 = time.time()
        dur_time = t2-t1
        #环境的返回值
        return self.expert_nodes, self.expert_traf, self.gpu_nodes, self.gpu_links, self.mask_expert, self.mask_gpu, dur_time, gpu_done, rewards