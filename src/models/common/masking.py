import torch

def _validate(batch_size,num_tokens,ratio):
    if batch_size<=0: raise ValueError("batch_size must be > 0")
    if num_tokens<=0: raise ValueError("num_tokens must be > 0")
    if not 0<=ratio<=1: raise ValueError("ratio must be in [0, 1]")

def make_temporal_mask(batch_size:int,num_tokens:int,ratio:float,*,generator=None,device=None):
    _validate(batch_size,num_tokens,ratio); count=int(num_tokens*ratio + 0.5)  # explicit round-half-up
    mask=torch.zeros((batch_size,num_tokens),dtype=torch.bool,device=device)
    if count:
        for b in range(batch_size):
            perm=torch.randperm(num_tokens,generator=generator).to(device=device); mask[b,perm[:count]]=True
    return mask

def visible_from_mask(masked_mask:torch.Tensor):
    if masked_mask.dtype is not torch.bool: raise ValueError("masked_mask must be bool")
    return ~masked_mask

def select_tokens(tokens:torch.Tensor,pos_ids:torch.Tensor,selection_mask:torch.Tensor):
    """Batch-preserving selection. Every sample must select the same K tokens."""
    if tokens.ndim!=3 or selection_mask.ndim!=2: raise ValueError("expected tokens [B,N,D] and selection_mask [B,N]")
    b,n,_=tokens.shape
    if b<=0: raise ValueError("batch size B must be > 0")
    if n<=0: raise ValueError("token count N must be > 0")
    if selection_mask.shape!=(b,n): raise ValueError("selection_mask shape must be [B,N]")
    if selection_mask.dtype is not torch.bool: raise ValueError("selection_mask must be bool")
    if pos_ids.ndim==1: pos_ids=pos_ids.unsqueeze(0).expand(tokens.shape[0],-1)
    pos_ids=pos_ids.to(device=tokens.device)
    if pos_ids.shape!=selection_mask.shape: raise ValueError("pos_ids shape mismatch")
    counts=selection_mask.sum(dim=1)
    if not torch.all(counts==counts[0]): raise ValueError("selection must contain the same count per sample")
    k=int(counts[0]); indices=selection_mask.nonzero(as_tuple=False).view(tokens.shape[0],k,2)[...,1]
    return torch.gather(tokens,1,indices.unsqueeze(-1).expand(-1,-1,tokens.shape[-1])), torch.gather(pos_ids,1,indices)
