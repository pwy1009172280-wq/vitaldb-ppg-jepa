import torch, pytest
from src.models.common import BackboneConfig
from src.models.data2vec import Data2VecStyle1D
from src.models.data2vec.masking import make_temporal_span_mask
from src.pretrain.config import load_config
from src.pretrain.optim import adamw_parameter_groups

def model(**kw):
    args=dict(patch_size=10,mask_ratio=.5,mask_span_length=2,target_top_k_layers=2); args.update(kw); return Data2VecStyle1D(BackboneConfig(16,2,4,2),**args)
def test_forward_backward_and_teacher_isolation():
    m=model(); o=m(torch.randn(2,1,50),generator=torch.Generator().manual_seed(4)); assert torch.isfinite(o.loss) and o.prediction.shape==o.target.shape==(2,5,16); o.loss.backward(); assert all(p.grad is None and not p.requires_grad for p in list(m.teacher_patch.parameters())+list(m.teacher_encoder.parameters())); assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in m.student_encoder.parameters())
def test_teacher_initialized_equal_and_train_keeps_eval():
    m=model(); assert all(torch.equal(a,b) for a,b in zip(m.student_patch.parameters(),m.teacher_patch.parameters())); assert all(torch.equal(a,b) for a,b in zip(m.student_encoder.parameters(),m.teacher_encoder.parameters())); m.train(); assert m.student_encoder.training and not m.teacher_encoder.training and not m.teacher_patch.training
def test_target_is_full_unmasked_and_uniform_last_k_layer_normed():
    m=model(target_top_k_layers=1); x=torch.randn(1,1,50); target=m._teacher_target(x); t,p=m.teacher_patch(x); f=m.teacher_encoder(t,p,return_hidden_states=True); expected=torch.nn.functional.layer_norm(f.hidden_states[-1],(16,),weight=None,bias=None); assert torch.allclose(target,expected)
    m2=model(target_top_k_layers=2); t,p=m2.teacher_patch(x); f=m2.teacher_encoder(t,p,return_hidden_states=True); expected=torch.nn.functional.layer_norm(torch.stack(f.hidden_states[-2:]).mean(0),(16,),weight=None,bias=None); assert torch.allclose(m2._teacher_target(x),expected)
def test_temporal_span_mask_count_positions_and_reproducibility():
    a=make_temporal_span_mask(2,12,.5,3,generator=torch.Generator().manual_seed(8)); b=make_temporal_span_mask(2,12,.5,3,generator=torch.Generator().manual_seed(8)); assert torch.equal(a,b) and a.sum(1).tolist()==[6,6]
    for row in a:
        blocks=[row[i:i+3] for i in range(0,12,3)]; partial=sum(bool(b.any()) and not bool(b.all()) for b in blocks)
        assert partial<=1
def test_ema_and_optimizer_excludes_teacher():
    m=model(); old=next(m.teacher_encoder.parameters()).detach().clone(); student=next(m.student_encoder.parameters()); student.data.add_(1.); m.ema_update(.25); new=next(m.teacher_encoder.parameters()); assert torch.allclose(new,.25*old+.75*student)
    groups=adamw_parameter_groups(m,.1); ids={id(p) for g in groups for p in g['params']}; train={id(p) for p in m.parameters() if p.requires_grad}; frozen={id(p) for p in list(m.teacher_patch.parameters())+list(m.teacher_encoder.parameters())}; assert ids==train and not ids&frozen and id(m.mask_token) in ids and id(m.prediction_head.weight) in ids
def test_validation_and_config():
    with pytest.raises(ValueError): model(target_top_k_layers=3)
    with pytest.raises(ValueError): model(ema_momentum=1.1)
    with pytest.raises(ValueError): model(loss_beta=-1)
    c=load_config('configs/smoke/data2vec.yaml'); assert c.method=='data2vec' and c.data2vec.target_top_k_layers==2
    from src.pretrain.config import SSLConfig,ModelConfig,DataConfig,TrainConfig,MAEConfig,Data2VecConfig
    assert SSLConfig('mae',ModelConfig(depth=1),DataConfig(),TrainConfig(),MAEConfig(.5),Data2VecConfig(target_top_k_layers=2)).validate().method=='mae'
    with pytest.raises(ValueError): SSLConfig('data2vec',ModelConfig(depth=2),DataConfig(),TrainConfig(),MAEConfig(),Data2VecConfig(target_top_k_layers=3)).validate()
    with pytest.raises(ValueError): SSLConfig('mae',ModelConfig(),DataConfig(),TrainConfig(),MAEConfig(0),Data2VecConfig()).validate()
def test_cpu_mask_generator_contract():
    g=torch.Generator(device='cpu').manual_seed(3); assert torch.equal(make_temporal_span_mask(1,10,.5,2,generator=g),make_temporal_span_mask(1,10,.5,2,generator=torch.Generator(device='cpu').manual_seed(3)))

@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required for AMP regression')
def test_data2vec_cuda_amp_forward_backward():
    m=model().cuda(); x=torch.randn(2,1,50,device='cuda')
    with torch.autocast(device_type='cuda',dtype=torch.float16): o=m(x,generator=torch.Generator(device='cpu').manual_seed(1))
    assert torch.isfinite(o.loss); o.loss.backward(); assert m.mask_token.grad is not None and torch.isfinite(m.mask_token.grad).all(); assert all(p.grad is None for p in list(m.teacher_patch.parameters())+list(m.teacher_encoder.parameters()))
