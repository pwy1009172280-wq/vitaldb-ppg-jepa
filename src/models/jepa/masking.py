from dataclasses import dataclass
import torch

def _count(n,ratio,name):
    if not 0<=ratio<=1: raise ValueError(f"{name}_ratio must be in [0, 1]")
    return int(n*ratio + 0.5)  # explicit round-half-up

@dataclass
class JEPAMask:
    context_mask: torch.Tensor
    target_mask: torch.Tensor
    context_indices: torch.Tensor
    target_indices: torch.Tensor
    context_pos_ids: torch.Tensor
    target_pos_ids: torch.Tensor

def make_jepa_masks(batch_size:int,num_tokens:int,context_ratio:float,target_ratio:float,*,generator=None,device=None):
    if batch_size<=0: raise ValueError("batch_size must be > 0")
    if num_tokens<=0: raise ValueError("num_tokens must be > 0")
    kc=_count(num_tokens,context_ratio,"context"); kt=_count(num_tokens,target_ratio,"target")
    if kc+kt>num_tokens: raise ValueError(f"requested context ({kc}) + target ({kt}) tokens exceed {num_tokens}")
    ci=[]; ti=[]
    for _ in range(batch_size):
        perm=torch.randperm(num_tokens,generator=generator)
        ci.append(perm[:kc]); ti.append(perm[kc:kc+kt])
    context_indices=torch.stack(ci).to(device=device); target_indices=torch.stack(ti).to(device=device)
    context_mask=torch.zeros((batch_size,num_tokens),dtype=torch.bool,device=device); target_mask=torch.zeros_like(context_mask)
    if kc: context_mask.scatter_(1,context_indices,True)
    if kt: target_mask.scatter_(1,target_indices,True)
    ids=torch.arange(num_tokens,device=device,dtype=torch.long).unsqueeze(0).expand(batch_size,-1).contiguous()
    return JEPAMask(context_mask,target_mask,context_indices,target_indices,torch.gather(ids,1,context_indices),torch.gather(ids,1,target_indices))
