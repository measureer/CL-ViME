import torch
import torch.nn as nn
import torch.nn.functional as F


class MoCo(nn.Module):
    """
    Build a MoCo model with a base encoder, a momentum encoder, and two MLPs
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, dim=256, mlp_dim=4096, T=1.0):
        """
        dim: feature dimension (default: 256)
        mlp_dim: hidden dimension in MLPs (default: 4096)
        T: softmax temperature (default: 1.0)
        """
        super(MoCo, self).__init__()

        self.T = T
        self.num_experts = 4
        self.top_k = 2
        # build encoders
        self.base_encoder = base_encoder(num_classes=mlp_dim)
        self.momentum_encoder = base_encoder(num_classes=mlp_dim)

        self._build_projector_and_predictor_mlps(dim, mlp_dim)

        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data.copy_(param_b.data)  # initialize
            param_m.requires_grad = False  # not update by gradient

    def _build_mlp(self, num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
        mlp = []
        for l in range(num_layers):
            dim1 = input_dim if l == 0 else mlp_dim
            dim2 = output_dim if l == num_layers - 1 else mlp_dim

            mlp.append(nn.Linear(dim1, dim2, bias=False))

            if l < num_layers - 1:
                mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU(inplace=True))
            elif last_bn:
                # follow SimCLR's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
                # for simplicity, we further removed gamma in BN
                mlp.append(nn.BatchNorm1d(dim2, affine=False))

        return nn.Sequential(*mlp)

    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        pass

    @torch.no_grad()
    def _update_momentum_encoder(self, m):
        """Momentum update of the momentum encoder"""
        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data = param_m.data * m + param_b.data * (1. - m)

    def contrastive_loss(self, q, k):
        # normalize
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        # gather all targets
        k = concat_all_gather(k)
        # Einstein sum is more intuitive
        logits = torch.einsum('nc,mc->nm', [q, k]) / self.T
        N = logits.shape[0]  # batch size per GPU
        labels = (torch.arange(N, dtype=torch.long) + N * torch.distributed.get_rank()).cuda()
        return nn.CrossEntropyLoss()(logits, labels) * (2 * self.T)
    def moeforward(self, v_standard):
        # MoE processing
        # v_standard  # [B, embed_dim]
        routing_logits = self.router(v_standard)  # [B, num_experts]
        routing_weights = F.softmax(routing_logits, dim=-1)  # [B, num_experts]
        '''
        # 计算辅助损失（负载均衡损失）
        mean_probs = routing_weights.mean(dim=0)  # [num_experts]
        aux_loss = self.num_experts * torch.sum(mean_probs * mean_probs)
        '''
        # 2. Select top-k experts（维度调整）
        topk_weights, topk_indices = torch.topk(routing_weights, self.top_k, dim=-1)  # [B, top_k]
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        # 生成专家输出（调整stack维度）
        expert_outputs = torch.stack([expert(v_standard) for expert in self.projector], dim=1)  # [B, num_experts, expert_dim]

        # 调整索引逻辑（移除token维度）
        batch_indices = torch.arange(expert_outputs.size(0), device=expert_outputs.device)[:, None]
        
        # 新索引方式 [B, top_k, expert_dim]
        selected_outputs = expert_outputs[batch_indices, topk_indices]
        
        # 加权求和（调整维度匹配）
        moe_output = (selected_outputs * topk_weights.unsqueeze(-1)).sum(dim=1)  # [B, expert_dim]
        return moe_output, routing_weights
    def forward(self, x1, x2, m,lamda):
        """
        Input:
            x1: first views of images
            x2: second views of images
            m: moco momentum
        Output:
            loss
        """
        q1_moe, q1_routing = self.moeforward(self.base_encoder(x1))
        q2_moe, q2_routing = self.moeforward(self.base_encoder(x2))
        q1_p = self.predictor(q1_moe)
        q2_p = self.predictor(q2_moe)
        with torch.no_grad():  # no gradient
            self._update_momentum_encoder(m)  # update the momentum encoder

            # compute momentum features as targets
            k1_moe,k1_routing = self.moeforward(self.momentum_encoder(x1))
            k2_moe,k2_routing = self.moeforward(self.momentum_encoder(x2))

        moe_loss = (F.cosine_similarity(q1_routing, q2_routing.detach()).mean()+F.cosine_similarity(q2_routing, q1_routing.detach()).mean()-F.cosine_similarity(q1_routing, k1_routing).mean()-F.cosine_similarity(q2_routing, k2_routing).mean())*0.5
        #moe_loss=self.contrastive_loss(q1_routing,q2_routing.detach())+self.contrastive_loss(q2_routing,q1_routing.detach())
        contrastive_loss = self.contrastive_loss(q1_p, k2_moe) + self.contrastive_loss(q2_p, k1_moe)
        return contrastive_loss +moe_loss*lamda


class MoCo_ResNet(MoCo):
    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        hidden_dim = self.base_encoder.fc.weight.shape[1]
        del self.base_encoder.fc, self.momentum_encoder.fc # remove original fc layer

        # projectors
        self.base_encoder.fc = self._build_mlp(2, hidden_dim, mlp_dim, dim)
        self.momentum_encoder.fc = self._build_mlp(2, hidden_dim, mlp_dim, dim)

        # predictor
        self.predictor = self._build_mlp(2, dim, mlp_dim, dim, False)


class MoCo_ViT(MoCo):
    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        input_dim = self.base_encoder.embed_dim
        #del self.base_encoder.head # remove original fc layer

        # 引入非线性层和残差连接
        self.router = nn.Sequential(
            nn.Linear(input_dim,self.num_experts),
            nn.LayerNorm(self.num_experts),
            #nn.ReLU()
        )


        # MoeProjectors
        self.projector = nn.ModuleList([
            self._build_mlp(3, input_dim, mlp_dim, dim) for _ in range(self.num_experts)
        ])
        
        # predictor
        self.predictor = self._build_mlp(2, dim, mlp_dim, dim)


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output