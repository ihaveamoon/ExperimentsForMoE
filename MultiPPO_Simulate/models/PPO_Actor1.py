import torch.nn as nn
from models.mlp import MLPActor
from models.mlp import MLPCritic,MLP
import torch.nn.functional as F
from models.graphCNN import GraphCNN
from torch.distributions.categorical import Categorical
import torch
from Params import configs
from Mhattention import ProbAttention
from agent_utils import greedy_select_action, select_gpus
from models.Pointer import Pointer
INIT = configs.Init
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.W1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W2 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.vt = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, decoder_state, encoder_outputs):
        # (batch_size, max_seq_len, hidden_size)
        encoder_transform = self.W1(encoder_outputs)

        # (batch_size, 1 (unsqueezed), hidden_size)
        decoder_transform = self.W2(decoder_state).unsqueeze(1)

        # 1st line of Eq.(3) in the paper
        # (batch_size, max_seq_len, 1) => (batch_size, max_seq_len)
        u_i = self.vt(torch.tanh(encoder_transform + decoder_transform)).squeeze(-1)
        return u_i

class Encoder(nn.Module):
    def __init__(self,num_layers, num_mlp_layers, input_dim,  hidden_dim, learn_eps, neighbor_pooling_type, device):
        super(Encoder,self).__init__()
        self.feature_extract = GraphCNN(num_layers=num_layers,
                                        num_mlp_layers=num_mlp_layers,
                                        input_dim=input_dim,
                                        hidden_dim=hidden_dim,
                                        learn_eps=learn_eps,
                                        neighbor_pooling_type=neighbor_pooling_type,
                                        device=device).to(device)
    def forward(self, expert_nodes, graph_pool, padded_nei, expert_links,):
        h_pooled, h_nodes = self.feature_extract(expert_nodes=expert_nodes,
                                                 graph_pool=graph_pool,
                                                 padded_nei=padded_nei,
                                                 expert_links=expert_links)

        return h_pooled,h_nodes

class Expert_Actor(nn.Module):
    def __init__(self,
                 n_moe_layer,
                 n_e,
                 num_layers,
                 learn_eps,
                 neighbor_pooling_type,
                 input_dim,
                 hidden_dim,
                 num_mlp_layers_feature_extract,
                 num_mlp_layers_critic,
                 hidden_dim_critic,
                 device
                 ):
        super(Expert_Actor, self).__init__()
        # expert_select size for problems
        self.n_moe_layer = n_moe_layer
        self.n_e = n_e
        self.device = device
        self.bn = torch.nn.BatchNorm1d(input_dim).to(device)
        # gpu size for problems
        self.device = device
        # network setup
        self.encoder = Encoder(num_layers=num_layers,
                               num_mlp_layers=num_mlp_layers_feature_extract,
                               input_dim=input_dim,
                               hidden_dim=hidden_dim,
                               learn_eps=learn_eps,
                               neighbor_pooling_type=neighbor_pooling_type,
                               device=device).to(device)
        self._input = nn.Parameter(torch.Tensor(hidden_dim))
        self._input.data.uniform_(-1, 1).to(device)
        self.actor1 = MLPActor(3, hidden_dim * 3, hidden_dim, 1).to(device)
        self.critic = MLPCritic(num_mlp_layers_critic, hidden_dim, hidden_dim_critic, 1).to(device)
        if INIT:
            for name, p in self.named_parameters():
                if 'weight' in name:
                    if len(p.size()) >= 2:
                        nn.init.orthogonal_(p, gain=1)
                elif 'bias' in name:
                    nn.init.constant_(p, 0)

    def forward(self, 
                expert_nodes, 
                expert_links, 
                graph_pool,
                padded_nei, 
                expert_candidate, 
                mask_expert, 
                old_policy=True,
                T=1,
                greedy=True
                ):
        print("\nExpert_Actor Forward:\n")
        h_pooled, h_nodes = self.encoder(expert_nodes=expert_nodes,
                                         graph_pool=graph_pool,
                                         padded_nei=padded_nei,
                                         expert_links=expert_links)
        print("\nExpert_Actor encoder success!")
        print("h_pooled: shape ", h_pooled.shape, "\n")
        print("h_nodes: shape ", h_nodes.shape, "\n")
        if old_policy:
            # Prepare features for actor
            dummy = expert_candidate.unsqueeze(-1).expand(-1, 1, h_nodes.size(-1)) 
            candidate_feature = torch.gather(h_nodes, 1, dummy)
            expert_context = torch.cat((candidate_feature, h_pooled.unsqueeze(1).expand_as(candidate_feature)), dim=-1)
            
            # 通过 MLP 网络计算得到分数
            candidate_scores = self.actor1(expert_context).squeeze()
            candidate_scores = candidate_scores - candidate_scores.max()  # Stabilize with log-sum-exp trick
            candidate_probs = F.softmax(candidate_scores / T, dim=1)

            if greedy:
                action, indices = torch.max(candidate_probs, dim=1)
                print("Greedy select expert: \n", action, " ,", indices, "\n")
            else:
                dist = torch.distributions.Categorical(candidate_probs)
                indices = dist.sample()
                log_e = dist.log_prob(indices)
            # Select features and links for output
            expert_index = expert_candidate.gather(1, indices.view(-1, 1)).squeeze(-1)
            expert_feature = candidate_feature.gather(1, indices.view(-1, 1, 1)).squeeze(1)
            expert_link = expert_links.gather(1, indices.view(-1, 1, 1).expand(-1, -1, self.n_e)).squeeze(1)

            print("Selected expert indices:", expert_index, "\n")
            print("Selected expert features:", expert_feature, "\n")
            print("Selected expert link features:", expert_link, "\n")

            return expert_index, expert_feature, expert_link, log_e

        else:
            expert_candidate_idx = expert_candidate.unsqueeze(-1).expand(-1, self.n_e, h_nodes.size(-1))
            batch_node = h_nodes.reshape(expert_candidate_idx.size(0), -1, expert_candidate_idx.size(-1)).to(self.device)
            candidate_feature = torch.gather(h_nodes.reshape(expert_candidate_idx.size(0), -1, expert_candidate_idx.size(-1)), 1, expert_candidate_idx)

            h_pooled_repeated = h_pooled.unsqueeze(-2).expand_as(candidate_feature)
            if gpu_pool == None:
                gpu_pooled_repeated = self._input[None, None, :].expand_as(candidate_feature).to(self.device)
            else:
                gpu_pooled_repeated = gpu_pool.unsqueeze(-2).expand_as(candidate_feature).to(self.device)
            concateFea = torch.cat((candidate_feature, h_pooled_repeated, gpu_pooled_repeated), dim=-1)
            candidate_scores = self.actor1(concateFea)

            candidate_scores = candidate_scores.squeeze(-1) * 10
            mask_reshape = mask_expert.reshape(candidate_scores.size())
            candidate_scores[mask_reshape] = float('-inf')

            pi = F.softmax(candidate_scores, dim=1)
            dist = Categorical(pi)

            log_e = dist.log_prob(e_index.to(self.device))
            entropy = dist.entropy()

            action1 = old_expert.type(torch.long).cuda()
            mask_gpu = mask_gpu.reshape(expert_candidate_idx.size(0), -1, self.n_g)
            mask_gpu_action = torch.gather(mask_gpu, 1,
                                           action1.unsqueeze(-1).unsqueeze(-1).expand(mask_gpu.size(0), -1,
                                                                                      mask_gpu.size(2)))
            # --------------------------------------------------------------------------------------------------------------------
            expert_node = torch.gather(batch_node, 1,
                                          action1.unsqueeze(-1).unsqueeze(-1).expand(batch_node.size(0), -1,
                                                                                     batch_node.size(2))).squeeze(1)
            v = self.critic(h_pooled)

            return entropy, v, log_e, expert_node.detach(), mask_gpu_action.detach(), h_pooled.detach()


class GPU_Actor(nn.Module):
    def __init__(self, input_dim_expert, input_dim_gpu, n_g, n_e, hidden_size, device):
        super(GPU_Actor, self).__init__()
        self.n_g = n_g
        self.n_e = n_e
        self.device = device
        self.hidden_size = hidden_size

        # Transformations for expert and GPU node features
        self.expert_transform = nn.Linear(input_dim_expert, hidden_size, bias=False).to(device)
        self.gpu_transform = nn.Linear(input_dim_gpu, hidden_size, bias=False).to(device)

        # Attention layers to integrate link features (expert-to-expert and gpu-to-gpu)
        self.expert_attention = nn.Linear(hidden_size * 2 + 1, 1, bias=False).to(device)  # +1 for expert_affinity
        self.gpu_attention = nn.Linear(hidden_size * 2 + 2, 1, bias=False).to(device)  # +2 for bandwidth and traffic

        # Actor head for outputting probabilities
        self.actor = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size, bias=False),  # combines transformed features
            nn.ReLU(),
            nn.Linear(hidden_size, n_g, bias=False)  # outputs a score for each GPU
        ).to(device)

    def forward(self, expert_node, gpu_nodes, expert_links, gpu_links, mask_gpu_action):
        # Transform node features
        expert_node = self.expert_transform(expert_node)
        gpu_nodes = self.gpu_transform(gpu_nodes)

        # Process link features with attention
        expert_att_input = torch.cat([
            expert_node.unsqueeze(2).expand(-1, -1, self.n_e, -1),
            expert_node.unsqueeze(1).expand(-1, self.n_e, -1, -1),
            expert_links['affinity'].unsqueeze(-1)], dim=-1)  # Including expert_affinity in attention
        expert_context = torch.sum(self.expert_attention(expert_att_input), dim=2)

        gpu_att_input = torch.cat([
            gpu_nodes.unsqueeze(2).expand(-1, -1, self.n_g, -1),
            gpu_nodes.unsqueeze(1).expand(-1, self.n_g, -1, -1),
            gpu_links['bandwidth'].unsqueeze(-1),
            gpu_links['traffic'].unsqueeze(-1)], dim=-1)  # Including bandwidth and traffic in attention
        gpu_context = torch.sum(self.gpu_attention(gpu_att_input), dim=2)

        # Combine node features with contextual information
        combined_features = torch.cat([expert_context, gpu_context], dim=-1)

        # Compute probabilities
        gpu_scores = self.actor(combined_features)
        gpu_scores = gpu_scores.masked_fill(mask_gpu_action, float('-inf'))  # Apply mask
        gpu_probabilities = F.softmax(gpu_scores, dim=-1)

        return gpu_probabilities


if __name__ == '__main__':
    print('Go home')
