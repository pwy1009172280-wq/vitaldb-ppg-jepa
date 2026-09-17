import torch
def make_temporal_span_mask(batch_size,num_tokens,ratio,span_length,*,generator=None,device=None):
    if batch_size<=0 or num_tokens<=0: raise ValueError('batch_size and num_tokens must be > 0')
    if not 0<ratio<1: raise ValueError('mask_ratio must be in (0,1)')
    if span_length<=0: raise ValueError('span_length must be > 0')
    if generator is not None and generator.device.type!='cpu': raise ValueError('formal temporal-mask generator must be a CPU torch.Generator')
    count=int(num_tokens*ratio+0.5)
    if not 0<count<num_tokens: raise ValueError(f'invalid mask count: mask_ratio={ratio}, num_tokens={num_tokens}, effective_masked_count={count}')
    out=torch.zeros((batch_size,num_tokens),dtype=torch.bool,device=device)
    for b in range(batch_size):
        blocks=[torch.arange(start,min(start+span_length,num_tokens)) for start in range(0,num_tokens,span_length)]
        remaining=count
        for block_id in torch.randperm(len(blocks),generator=generator):
            block=blocks[int(block_id)]; take=min(remaining,int(block.numel())); out[b,block[:take].to(device=out.device)]=True; remaining-=take
            if not remaining: break
    return out
