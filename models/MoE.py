import torch
import torch.nn as nn
import torch.nn.functional as F

# We can make this way more efficient by stacking the weights of the experts
# removes looping in the forward MoE call - can optimize later.


class Expert(nn.Module):
    def __init__(self,embed_dim, mlp_scale : int = 2, drop_rate: float = 0.0):
        super().__init__()
        self.nn = nn.Sequential(*[nn.Linear(embed_dim,embed_dim * mlp_scale),nn.GELU(),nn.Linear(embed_dim * mlp_scale,embed_dim),nn.Dropout(drop_rate)])

    def forward(self,x):
        return self.nn(x)


class Router(nn.Module):
    def __init__(self,embed_dim,num_experts=4,num_classes=2,epsilon=0.01,
                     freeze_old_classes=False, old_num_classes=None,device='cuda'):
        super().__init__()
        assert num_experts % num_classes == 0
        self.device = device
        self.epsilon = 0.1
        self.freeze_old_classes = freeze_old_classes
        self.old_num_classes = old_num_classes
        # Supervised routing - split the experts into all materials
        # We enforce the load to be balanced
        self.k = num_experts // num_classes # experts per class 
        targets = [i for i in range(num_classes) for _ in range(num_experts // num_classes)]
        self.expert_index = torch.tensor(targets,dtype=torch.long).to(self.device)
        self.router = nn.Linear(embed_dim,num_experts)

    def mask_old_expert_grads(self,grad):
        grad_masked = grad.clone()
        num_old_experts = self.old_num_classes * self.k
        grad_masked[:, :, :num_old_experts] = 0.0
        return grad_masked
            
        
    def forward(self,x,class_label,padding_mask=None,return_indices=False):
        loss = None
        indices = None
        B,seq,embed_dim = x.shape
        expert_mask = (self.expert_index.unsqueeze(0) == class_label.unsqueeze(1))  # (B, num_experts)
        mask = ~expert_mask.unsqueeze(1) # (B, 1, num_experts)

        if return_indices:
            indices = [torch.where(m)[0] for m in expert_mask]  # list of length B

        routing = self.router(x)
        routing.masked_fill_(mask,-torch.inf)

        weights = F.softmax(routing,dim=-1)

        # Triggered in fine tuning when adding new classes
        if (self.training and weights.requires_grad and
            self.freeze_old_classes and self.old_num_classes is not None):
            weights.register_hook(self.mask_old_expert_grads)

        if self.training:
            if self.k == 1:
                loss = torch.tensor(0.0)
            else:
                # We need per class expert balancing
                target_load = expert_mask[:, None, :].expand(-1, x.size(1), -1).float() / self.k # (B,seq,num_experts)
                load = weights # (B,seq,num_experts)

                if padding_mask is not None:
                    valid_mask = (~padding_mask).float()  
                    valid_mask_exp = valid_mask.unsqueeze(-1)  
                    
                    diff = (target_load - load) ** 2

                    loss = (diff * valid_mask_exp).sum() / valid_mask_exp.sum()
                else:
                    loss = F.mse(target_load,load)

        return weights,indices,loss



class MoE(nn.Module):
    def __init__(self,embed_dim,mlp_scale=2,drop_rate=0.0,num_experts=4,num_classes=2,device='cuda'):
        super().__init__()
        self.embed_dim = embed_dim
        self.mlp_scale = mlp_scale
        self.drop_rate = drop_rate
        self.num_experts = num_experts
        self.num_classes = num_classes
        self.device = device

        self.router = Router(self.embed_dim,num_experts=self.num_experts,num_classes=self.num_classes)
        self.experts = nn.ModuleList([Expert(embed_dim, mlp_scale, drop_rate) for _ in range(self.num_experts)])

    def forward(self,x,class_label,padding_mask=None):
        B,seq_len,embed_dim = x.shape
        load_balance = None
        weights,indices,load_balance = self.router(x,class_label,padding_mask=padding_mask) # (B,seq_len,num_experts),list -> len = batch, indices[0] = (2,)
        
        output = torch.zeros(x.shape,device=x.device) # (B,seq_len,embed_dim)

        for i,expert in enumerate(self.experts):
            # weights of shape (B,seq_len,)->(B,seq_len,1) for element-wise multiplication
            # expert (B,seq_len,embed_dim) 
            output += weights[:,:,i].unsqueeze(-1) * expert(x)

        return output,load_balance














