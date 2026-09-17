import torch,pytest
from src.models.common import BackboneConfig
from src.models.jepa import JEPA1D
from src.models.jepa.masking import make_target_block_masks
from src.pretrain.config import load_config
from src.pretrain.optim import adamw_parameter_groups
def model(**kw):
    a=dict(patch_size=10,patch_stride=10,num_target_blocks=2,target_block_length=1,predictor_dim=8,predictor_depth=1,predictor_num_heads=2); a.update(kw); return JEPA1D(BackboneConfig(8,1,2,2),**a)
def test_forward_backward_and_teacher_isolation():
    m=model(); o=m(torch.randn(2,1,50),generator=torch.Generator().manual_seed(2)); assert torch.isfinite(o.loss) and o.prediction.shape==o.target.shape==(2,2,8); o.loss.backward(); assert all(p.grad is None and not p.requires_grad for p in list(m.target_patch.parameters())+list(m.target_encoder.parameters())); assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in m.predictor.parameters())
def test_init_train_eval_and_ema():
    m=model(); assert all(torch.equal(a,b) for a,b in zip(m.online_patch.parameters(),m.target_patch.parameters())); m.train(); assert m.context_encoder.training and m.predictor.training and not m.target_encoder.training; old=next(m.target_encoder.parameters()).detach().clone(); p=next(m.context_encoder.parameters()); p.data.add_(1); m.ema_update(.25); assert torch.allclose(next(m.target_encoder.parameters()),.25*old+.75*p)
def test_mask_geometry_and_rng():
    a=make_target_block_masks(2,12,2,2,generator=torch.Generator().manual_seed(4)); b=make_target_block_masks(2,12,2,2,generator=torch.Generator().manual_seed(4)); assert torch.equal(a.target_mask,b.target_mask) and torch.equal(a.context_mask,~a.target_mask) and not (a.context_mask&a.target_mask).any() and a.target_mask.sum(1).tolist()==[4,4]; assert torch.equal(a.context_pos_ids,a.context_indices) and torch.equal(a.target_pos_ids,a.target_indices)
    with pytest.raises(ValueError): make_target_block_masks(1,4,2,2)
def test_patch_config_context_and_optimizer():
    with pytest.raises(ValueError): model(patch_stride=5)
    m=model(); o=m(torch.randn(1,1,50),generator=torch.Generator().manual_seed(1)); assert torch.equal(o.context_mask|o.target_mask,torch.ones_like(o.target_mask)); groups=adamw_parameter_groups(m,.1); ids={id(p) for g in groups for p in g['params']}; train={id(p) for p in m.parameters() if p.requires_grad}; frozen={id(p) for p in list(m.target_patch.parameters())+list(m.target_encoder.parameters())}; assert ids==train and not ids&frozen and id(m.target_query) in ids and id(m.predictor_projection.weight) in ids and load_config('configs/smoke/jepa.yaml').method=='jepa'
def test_core_dataflow_hooks():
    m=model(); seen={}
    def hook(name):
        def f(_,args): seen[name]=(args[0].detach().clone(),args[1].detach().clone())
        return f
    hs=[m.context_encoder.register_forward_pre_hook(hook('context')),m.target_encoder.register_forward_pre_hook(hook('target')),m.predictor.register_forward_pre_hook(hook('predictor'))]
    x=torch.randn(1,1,50); patch,_=m.online_patch(x); o=m(x,generator=torch.Generator().manual_seed(5)); [h.remove() for h in hs]
    ct,cp=seen['context']; tt,tp=seen['target']; pt,pp=seen['predictor']; assert ct.shape[1]==5-o.target_count and tt.shape[1]==5 and pt.shape[1]==5; assert torch.equal(cp[0],o.context_mask[0].nonzero().flatten()) and torch.equal(tp[0],torch.arange(5)); assert torch.equal(pp[0],torch.arange(5)); assert torch.equal(pt[0,o.target_mask[0]],m.target_query[0].expand(o.target_count,-1)); assert not torch.equal(pt[0,o.target_mask[0]],patch[0,o.target_mask[0]])

@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required for AMP regression')
def test_jepa_cuda_amp_forward_backward():
    m=model().cuda(); x=torch.randn(2,1,50,device='cuda')
    with torch.autocast(device_type='cuda',dtype=torch.float16): o=m(x,generator=torch.Generator(device='cpu').manual_seed(1))
    assert torch.isfinite(o.loss); o.loss.backward(); assert m.target_query.grad is not None and torch.isfinite(m.target_query.grad).all(); assert all(p.grad is None for p in list(m.target_patch.parameters())+list(m.target_encoder.parameters()))
