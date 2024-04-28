import torch
import torch.nn as nn
import torch.nn.functional as F
from models.mlp import MLP
# import sys
# sys.path.append("models/")


class GraphCNN(nn.Module):
    def __init__(self,
                 num_layers,
                 num_mlp_layers,
                 input_dim,
                 hidden_dim,
                 learn_eps,
                 neighbor_pooling_type,
                 device):
        '''
        num_layers: number of layers in the neural networks (INCLUDING the input layer)
        num_mlp_layers: number of layers in mlps (EXCLUDING the input layer)
        input_dim: dimensionality of input features
        hidden_dim: dimensionality of hidden units at ALL layers
        output_dim: number of classes for prediction
        learn_eps: If True, learn epsilon to distinguish center nodes from neighboring nodes. If False, aggregate neighbors and center nodes altogether.
        neighbor_pooling_type: how to aggregate neighbors (mean, average, or max)
        device: which device to use
        '''

        super(GraphCNN, self).__init__()

        self.device = device
        self.num_layers = num_layers
        self.neighbor_pooling_type = neighbor_pooling_type
        self.learn_eps = learn_eps

        # List of MLPs
        self.mlps = torch.nn.ModuleList()
        self.bn = torch.nn.BatchNorm1d(input_dim)

        # List of batchnorms applied to the output of MLP (input of the final prediction linear layer)
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(self.num_layers - 1):
            if layer == 0:
                self.mlps.append(MLP(num_mlp_layers, input_dim, hidden_dim))
            else:
                self.mlps.append(MLP(num_mlp_layers, hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

    def next_layer_eps(self, node_feature, layer, padded_neighbor_list = None, Adj_block = None):
        if self.neighbor_pooling_type == "max":
            pooled = self.maxpool(node_feature, padded_neighbor_list)
        else:
            pooled = torch.mm(Adj_block, node_feature) # 矩阵乘法，聚合邻居的特征
            if self.neighbor_pooling_type == "average":
                # If average pooling
                degree = torch.mm(Adj_block, torch.ones((Adj_block.shape[0], 1)).to(self.device))
                pooled = pooled/degree
        # Reweights the center node representation when aggregating it with its neighbors
        pooled = pooled + (1 + self.eps[layer]) * node_feature
        pooled_rep = self.mlps[layer](pooled) # 特征处理
        node_feature = self.batch_norms[layer](pooled_rep)
        node_feature = F.relu(node_feature)
        return node_feature


    def next_layer(self, node_feature, layer, padded_neighbor_list = None, Adj_block = None):
        # Calculate the degree of each node (sum of connections along the row)
        degree = Adj_block.sum(dim=2, keepdim=True)  # [B, N, 1]

        # Add a small epsilon to avoid division by zero
        epsilon = 1e-5
        Adj_block_normalized = Adj_block / (degree + epsilon)  # [B, N, N]

        # Pooling: Use normalized adjacency matrix to compute the new features
        pooled = torch.bmm(Adj_block_normalized, node_feature)  # [B, N, F]

        # Apply MLP and batch normalization
        pooled_rep = self.mlps[layer](pooled)
        node_feature = self.batch_norms[layer](pooled_rep)

        # Apply non-linearity
        node_feature = F.relu(node_feature)

        return node_feature


    def forward(self,
                expert_nodes,
                graph_pool,
                padded_nei,
                expert_links):

        node_feature = expert_nodes  # [64, 32, 2]
        Adj_block = expert_links # [64, 32, 32]
        for layer in range(self.num_layers-1):
            node_feature = self.next_layer(node_feature, layer, Adj_block=Adj_block) # [64, 32, 2]

        h_nodes = node_feature.clone()
        print(graph_pool.shape, node_feature.shape)
        pooled_h = torch.sparse.mm(graph_pool, node_feature) # 将节点级别的特征汇总成图级别的特征

        return pooled_h, h_nodes


if __name__ == '__main__':

    ''' Test attention block
    attention = Attention()
    g = torch.tensor([[1., 2.]], requires_grad=True)
    candidates = torch.tensor([[3., 3.],
                               [2., 2.]], requires_grad=True)

    ret = attention(g, candidates)
    print(ret)
    loss = ret.sum()
    print(loss)

    grad = torch.autograd.grad(loss, g)

    print(grad)
    '''