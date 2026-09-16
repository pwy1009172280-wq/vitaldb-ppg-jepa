import torch, pytest
from src.models.common.masking import make_temporal_mask,select_tokens

def test_mask_shape_count_and_edges():
 g=torch.Generator().manual_seed(4); m=make_temporal_mask(2,10,.3,generator=g); assert m.dtype==torch.bool and m.sum(1).tolist()==[3,3]; assert make_temporal_mask(2,10,0).sum()==0; assert make_temporal_mask(2,10,1).sum()==20
 with pytest.raises(ValueError): make_temporal_mask(0,10,.2)

def test_round_half_up():
 assert make_temporal_mask(1,5,.5).sum().item()==3  # 2.5 -> 3
 assert make_temporal_mask(1,7,.5).sum().item()==4  # 3.5 -> 4

def test_batch_selection_and_empty():
 x=torch.arange(2*6*3).view(2,6,3); ids=torch.arange(6); m=torch.tensor([[1,0,1,0,0,1],[0,1,0,1,0,1]],dtype=torch.bool); y,p=select_tokens(x,ids,m); assert y.shape==(2,3,3) and p.tolist()==[[0,2,5],[1,3,5]]
 z,q=select_tokens(x,ids,torch.zeros(2,6,dtype=torch.bool)); assert z.shape==(2,0,3) and q.shape==(2,0)
 with pytest.raises(ValueError): select_tokens(torch.empty(0,6,3),torch.arange(6),torch.empty(0,6,dtype=torch.bool))
 with pytest.raises(ValueError): select_tokens(torch.empty(2,0,3),torch.empty(0,dtype=torch.long),torch.empty(2,0,dtype=torch.bool))

def test_cuda_cpu_pos_ids():
 if not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
 x=torch.randn(2,6,4,device='cuda'); m=torch.tensor([[1,0,1,0,0,1],[0,1,0,1,0,1]],device='cuda',dtype=torch.bool); _,p=select_tokens(x,torch.arange(6),m); assert p.device.type=='cuda'
