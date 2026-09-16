import torch, pytest
from src.models.jepa.masking import make_jepa_masks

def test_jepa_exact_disjoint_batch_and_reproducible():
 a=make_jepa_masks(3,10,.3,.4,generator=torch.Generator().manual_seed(5)); b=make_jepa_masks(3,10,.3,.4,generator=torch.Generator().manual_seed(5)); assert torch.equal(a.context_indices,b.context_indices); assert torch.equal(a.target_indices,b.target_indices); assert a.context_mask.sum(1).tolist()==[3]*3 and a.target_mask.sum(1).tolist()==[4]*3; assert not (a.context_mask&a.target_mask).any(); assert a.context_indices.shape==(3,3) and a.target_indices.shape==(3,4) and a.context_pos_ids.shape==(3,3) and a.target_pos_ids.shape==(3,4); assert torch.equal(a.context_pos_ids,a.context_indices) and torch.equal(a.target_pos_ids,a.target_indices); assert any(not torch.equal(a.context_indices[0],a.context_indices[i]) for i in (1,2))

def test_jepa_rounding_edges_and_impossible():
 z=make_jepa_masks(2,5,.5,0); assert z.context_indices.shape==(2,3); z=make_jepa_masks(2,7,0,.5); assert z.target_indices.shape==(2,4); z=make_jepa_masks(2,4,0,1); assert z.context_indices.shape==(2,0) and z.target_indices.shape==(2,4)
 with pytest.raises(ValueError): make_jepa_masks(1,4,.75,.5)
 with pytest.raises(ValueError): make_jepa_masks(1,4,-.1,.2)
 with pytest.raises(ValueError): make_jepa_masks(0,4,.2,.2)
