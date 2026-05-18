import torch
import torch.nn as nn
import torch.nn.functional as F

class AGLayer(nn.Module):
    def __init__(self, args, num_op):
        super().__init__()
        self.args = args
        # 定义操作（op）的embedding，维度为 (num_op, graph_dim)
        # num_op：操作/算子数量
        # graph_dim：图的表示维度
        self.op_emb = nn.Embedding(num_op, args.graph_dim) # op * g_d

    def forward(self, g):
        # g: graph * g_d
        o = self.op_emb.weight
        o = o / o.norm(2, dim = -1, keepdim = True)
        cosloss = (o @ o.t()).sum()
        alpha = g @ o.t()
        alpha = alpha / self.args.temperature
        alpha = F.softmax(alpha, dim = 1) # graph * op
        alpha = alpha * (alpha > 1/6)
        alpha = alpha / alpha.sum(dim = 1, keepdim = True)
        return alpha, cosloss

class AG(nn.Module):
    def __init__(self, args, num_op, num_pool):
        super().__init__()
        self.args = args
        self.layers = nn.ModuleList()
        self.set = 'train'
        for i in range(args.num_layers):
            self.layers.append(AGLayer(args, num_op))

    def forward(self, g):
        alpha_all = []
        cosloss = torch.zeros(1).to(self.layers[0].op_emb.weight.device)

        for i in range(self.args.num_layers):
            alpha, closs = self.layers[i](g)
            cosloss = cosloss + closs
            alpha_all.append(alpha)

        return alpha_all, cosloss

class InvDisenHead(nn.Module): 
    def __init__(self, input_dim=8, invariant_dim=2, variant_dim=6):
        super().__init__()
        self.proj0 = nn.Linear(input_dim, invariant_dim)  # graph_emb0 -> 2D
        self.proj1 = nn.Linear(input_dim, variant_dim)  # graph_emb1 -> 6D

    def _cross_subspace_decorrelation(self, z, num_groups=3):
        bsz, dim = z.shape
        sub_dim = dim // num_groups
        parts = torch.chunk(z, num_groups, dim=-1)  # list of [batch_size, sub_dim]

        losses = []
        for i in range(num_groups):
            for j in range(i + 1, num_groups):
                p_i = parts[i]
                p_j = parts[j]

                p_i = (p_i - p_i.mean(-1, keepdim=True)) / (p_i.std(-1, keepdim=True) + 1e-6)
                p_j = (p_j - p_j.mean(-1, keepdim=True)) / (p_j.std(-1, keepdim=True) + 1e-6)

                corr = torch.einsum("bi,bj->bij", p_i, p_j) / p_i.size(1)

                loss_ij = (corr ** 2).mean(dim=(1, 2))  # [batch_size]
                losses.append(loss_ij)

        loss = torch.stack(losses, dim=0).mean(0)
        return loss.mean()

    def forward(self, graph_emb0, graph_emb1):
        z0 = self.proj0(graph_emb0)  # [batch_size, 2]
        z1 = self.proj1(graph_emb1)  # [batch_size, 6]

        loss = self._cross_subspace_decorrelation(z1, num_groups=3)

        out = torch.cat([z0, z1], dim=-1)
        return out, loss
