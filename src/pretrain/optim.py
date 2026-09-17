from torch import nn
def adamw_parameter_groups(module:nn.Module, weight_decay:float):
    decay=[]; no_decay=[]
    for owner in module.modules():
        for name,p in owner.named_parameters(recurse=False):
            if not p.requires_grad: continue
            if name=='bias' or p.ndim<=1 or isinstance(owner,(nn.LayerNorm,nn.BatchNorm1d,nn.BatchNorm2d,nn.GroupNorm)): no_decay.append(p)
            else: decay.append(p)
    return [{'params':decay,'weight_decay':weight_decay},{'params':no_decay,'weight_decay':0.}]
