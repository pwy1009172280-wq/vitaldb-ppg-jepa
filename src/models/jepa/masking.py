from dataclasses import dataclass
import torch
@dataclass
class JEPAMask:
    context_mask:torch.Tensor; target_mask:torch.Tensor; context_indices:torch.Tensor; target_indices:torch.Tensor; context_pos_ids:torch.Tensor; target_pos_ids:torch.Tensor
def make_target_block_masks(batch_size,num_tokens,num_target_blocks,target_block_length,*,generator=None,device=None):
    if batch_size<=0 or num_tokens<=0 or num_target_blocks<=0 or target_block_length<=0: raise ValueError('batch size, token count, block count, and block length must be > 0')
    total=num_target_blocks*target_block_length
    if total>=num_tokens: raise ValueError(f'JEPA target blocks require at least one context token: num_tokens={num_tokens}, requested_target_tokens={total}')
    if generator is not None and generator.device.type!='cpu': raise ValueError('formal JEPA temporal-mask generator must be a CPU torch.Generator')
    slack=num_tokens-total; all_masks=[]; all_indices=[]
    for _ in range(batch_size):
        cuts=torch.sort(torch.randperm(slack+num_target_blocks,generator=generator)[:num_target_blocks])[0]; gaps=torch.diff(torch.cat((torch.tensor([-1]),cuts,torch.tensor([slack+num_target_blocks-1]))))-1; starts=[]; cursor=int(gaps[0])
        for j in range(num_target_blocks): starts.append(cursor); cursor+=target_block_length+int(gaps[j+1])
        all_indices.append(torch.cat([torch.arange(s,s+target_block_length) for s in starts]))
    ti=torch.stack(all_indices).to(device=device); tm=torch.zeros((batch_size,num_tokens),dtype=torch.bool,device=device); tm.scatter_(1,ti,True); cm=~tm; ids=torch.arange(num_tokens,device=device).expand(batch_size,-1); ci=cm.nonzero().view(batch_size,num_tokens-total,2)[...,1]; return JEPAMask(cm,tm,ci,ti,torch.gather(ids,1,ci),ti)

# Retained for the existing common-mask regression; formal JEPA uses the
# target-block sampler above.
def make_jepa_masks(batch_size,num_tokens,context_ratio,target_ratio,*,generator=None,device=None):
    if batch_size<=0 or num_tokens<=0: raise ValueError('batch_size and num_tokens must be > 0')
    if not 0<=context_ratio<=1 or not 0<=target_ratio<=1: raise ValueError('ratios must be in [0, 1]')
    kc=int(num_tokens*context_ratio+0.5); kt=int(num_tokens*target_ratio+0.5)
    if kc+kt>num_tokens: raise ValueError('requested masks exceed token count')
    ci=[]; ti=[]
    for _ in range(batch_size):
        perm=torch.randperm(num_tokens,generator=generator); ci.append(perm[:kc]); ti.append(perm[kc:kc+kt])
    c=torch.stack(ci).to(device=device); t=torch.stack(ti).to(device=device); cm=torch.zeros((batch_size,num_tokens),dtype=torch.bool,device=device); tm=torch.zeros_like(cm); cm.scatter_(1,c,True); tm.scatter_(1,t,True); ids=torch.arange(num_tokens,device=device).expand(batch_size,-1); return JEPAMask(cm,tm,c,t,torch.gather(ids,1,c),torch.gather(ids,1,t))
