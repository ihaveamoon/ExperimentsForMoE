import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import torch
from Params import configs
from agent_utils import greedy_select_action, select_gpus
INIT = configs.Init

def initialize_weights(model):
    for name, p in model.named_parameters():
        if 'weight' in name:
            if len(p.size()) >= 2:
                nn.init.orthogonal_(p, gain=1)
        elif 'bias' in name:
            nn.init.constant_(p, 0)

class MLP(nn.Module):
    """ Multi-Layer Perceptron with a variable number of hidden layers """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList([nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim) for i in range(num_layers)])
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        if INIT:
            self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.layers:
            nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        nn.init.kaiming_uniform_(self.output_layer.weight, nonlinearity='relu')
        if self.output_layer.bias is not None:
            nn.init.zeros_(self.output_layer.bias)

    def forward(self, x):
        for layer in self.layers:
            x = self.relu(layer(x))
        return self.output_layer(x)

class MLPbm(nn.Module):
    """ Multi-Layer Perceptron with a variable number of hidden layers """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(MLPbm, self).__init__()
        # self.layers = nn.ModuleList([nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim) for i in range(num_layers)])
        # self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        if INIT:
            self._initialize_weights()
    
    def _initialize_weights(self):
        for layer in self.layers:
            nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        nn.init.kaiming_uniform_(self.output_layer.weight, nonlinearity='relu')
        if self.output_layer.bias is not None:
            nn.init.zeros_(self.output_layer.bias)

    def forward(self, x):
        # for layer in self.layers:
        #     x = self.relu(layer(x))
        for layer, batch_norm in zip(self.layers, self.batch_norms):
            x = self.relu(batch_norm(layer(x)))
        return self.output_layer(x)

class MLPtanh(nn.Module):
    """ Multi-Layer Perceptron with a variable number of hidden layers and Tanh activation """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(MLPtanh, self).__init__()
        # self.layers = nn.ModuleList([nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim) for i in range(num_layers)])
        # self.output_layer = nn.Linear(hidden_dim, output_dim)
        # self.tanh = nn.Tanh()
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.tanh = nn.Tanh()
        if INIT:
            self._initialize_weights()
    
    def _initialize_weights(self):
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(self.output_layer.weight)
        if self.output_layer.bias is not None:
            nn.init.zeros_(self.output_layer.bias)

    def forward(self, x):
        # for layer in self.layers:
        #     x = self.tanh(layer(x))
        for layer, batch_norm in zip(self.layers, self.batch_norms):
            x = self.tanh(batch_norm(layer(x)))
        return self.output_layer(x)

class GNNLayer(nn.Module):
    """ Single layer of Graph Neural Network """
    def __init__(self, feature_dim, eps, hidden_dim, output_dim, num_mlp_layers):
        super(GNNLayer, self).__init__()
        self.mlp = MLP(feature_dim, hidden_dim, output_dim, num_mlp_layers)
        self.eps = eps
        if INIT:
            initialize_weights(self)

    def forward(self, h, adj):
        # Aggregate neighbor features
        sum_neighbors = torch.bmm(adj, h)
        # Node feature update
        new_h = self.mlp((1 + self.eps) * h + sum_neighbors)
        return new_h

class GNN(nn.Module):
    """ Graph Neural Network consisting of multiple GNN layers """
    def __init__(self, feature_dim, num_layers, hidden_dim, output_dim, num_mlp_layers):
        super(GNN, self).__init__()
        self.layers = nn.ModuleList()
        eps_values = np.random.uniform(0.01, 0.1, num_layers)

        self.layers.append(GNNLayer(feature_dim, eps_values[0], hidden_dim, hidden_dim, num_mlp_layers))
        for i in range(1, num_layers - 1):
            self.layers.append(GNNLayer(hidden_dim, eps_values[i], hidden_dim, hidden_dim, num_mlp_layers))
        self.layers.append(GNNLayer(hidden_dim, eps_values[-1], hidden_dim, output_dim, num_mlp_layers))
        if INIT:
            initialize_weights(self)

    def forward(self, h, adj):
        for layer in self.layers:
            h = layer(h, adj)
        return h

class Expert_Encoder(nn.Module):
    """ Encoder for experts using GNN """
    def __init__(self, expert_feature_dim, hidden_dim, output_dim, num_layers, num_mlp_layers):
        super(Expert_Encoder, self).__init__()
        self.gnn = GNN(expert_feature_dim, num_layers, hidden_dim, output_dim, num_mlp_layers)
        if INIT:
            initialize_weights(self)

    def forward(self, node_features, adj_matrix):
        node_embeddings = self.gnn(node_features, adj_matrix)
        global_embedding = node_embeddings.mean(dim=1)
        return node_embeddings, global_embedding

class Expert_Decoder(nn.Module):
    """ Multi-Layer MLP Actor for decoding actions based on combined embeddings """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(Expert_Decoder, self).__init__()
        self.mlp = MLPbm(input_dim, hidden_dim, output_dim, num_layers)
        if INIT:
            initialize_weights(self)

    def forward(self, h_node, h_global,  mask):
        # print(f"Expert_Decoder forward: h_node.shape = {h_node.shape}, h_global.shape = {h_global.shape}, u.shape = {u.shape}")
        # Expand h_global and concatenate it with h_node and u
        h_global_expanded = h_global.unsqueeze(1).expand(-1, h_node.size(1), -1)
        h_combined = torch.cat([h_node, h_global_expanded], dim=-1)
        # print(f"h_combined.shape = {h_combined.shape}")

        batch_size, expert_num, feature_dim = h_combined.size()
        h_combined = h_combined.view(batch_size * expert_num, feature_dim)
        action_scores = self.mlp(h_combined).view(batch_size, expert_num, -1).squeeze(-1)
        
        # action_scores = self.mlp(h_combined).squeeze(-1)
        # Masking and softmax
        action_scores = action_scores.masked_fill(mask, float('-inf'))
        action_probs = F.softmax(action_scores, dim=1)
        return action_probs

class Expert_Actor(nn.Module):
    def __init__(self, encoder ,decoder):
        super(Expert_Actor, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        if INIT:
            initialize_weights(self)

    def forward(self, node_features, adj_matrix,  mask):
        # Decode to get action probabilities
        h_node,h_global = self.encoder(node_features,adj_matrix)
        action_probs = self.decoder(h_node, h_global,  mask)

        # Sampling or greedy action selection
        if self.training:  # Sampling strategy during training
            distribution = torch.distributions.Categorical(action_probs)
            action = distribution.sample()
        else:  # Greedy strategy during evaluation
            action = torch.argmax(action_probs, dim=1)

        return action_probs, action



class GPU_Encoder(nn.Module):
    """ Encoder for GPUs using GNN """
    def __init__(self, gpu_feature_dim, hidden_dim, output_dim, num_layers, num_mlp_layers):
        super(GPU_Encoder, self).__init__()
        self.gnn = GNN(gpu_feature_dim, num_layers, hidden_dim, output_dim, num_mlp_layers)
        if INIT:
            initialize_weights(self)

    def forward(self, gpu_nodes, gpu_links):
        node_embeddings = self.gnn(gpu_nodes, gpu_links)
        global_embedding = node_embeddings.mean(dim=1)
        return node_embeddings, global_embedding

class GPU_Decoder(nn.Module):
    """ Decoder to compute GPU action scores based on combined embeddings """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(GPU_Decoder, self).__init__()
        self.mlp = MLPtanh(input_dim, hidden_dim, output_dim, num_layers)
        if INIT:
            initialize_weights(self)

    def forward(self, h_node, h_global, mask):
        # print(f"GPU_Decoder forward: h_node[0] = {h_node[0]}, h_global[0] = {h_global}[0], u[0] = {u[0]}")
        # Expand h_global and concatenate it with h_node and u
        h_global_expanded = h_global.unsqueeze(1).expand(-1, h_node.size(1), -1)
        h_combined = torch.cat([h_node, h_global_expanded], dim=-1)
        # print(f"h_combined.shape = {h_combined.shape}")
        batch_size, gpu_num, feature_dim = h_combined.size()
        h_combined = h_combined.view(batch_size * gpu_num, feature_dim)
        action_scores = self.mlp(h_combined).view(batch_size, gpu_num, -1).squeeze(-1)
        
        # action_scores = self.mlp(h_combined).squeeze(-1)
        
        # Masking and softmax
        action_scores = action_scores.masked_fill(mask, float('-inf'))
        action_probs = F.softmax(action_scores, dim=1)
        return action_probs

class GPU_Actor(nn.Module):
    def __init__(self,encoder, decoder):
        super(GPU_Actor, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        if INIT:
            initialize_weights(self)

    def forward(self,gpu_nodes, gpu_links, mask):
        h_gpu,h_pooled_gpu = self.encoder(gpu_nodes,gpu_links)
        action_probs = self.decoder(h_gpu,  h_pooled_gpu, mask)
        # Determine expand/shrink actions
        gpu_actions = action_probs > 0.5
        return action_probs, gpu_actions



class MLPCritic(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        super(MLPCritic, self).__init__()
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(input_dim, hidden_dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.model = nn.Sequential(*layers)
        if INIT:
            initialize_weights(self)

    def forward(self, x):
        return self.model(x)