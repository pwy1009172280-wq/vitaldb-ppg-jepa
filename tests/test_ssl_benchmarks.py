import torch
import pytest
from src.models.ssl_benchmarks import build_method
@pytest.mark.parametrize('name',["mae","data2vec","jepa"])
def test_forward_backward_and_batch_shape(name):
 m=build_method(name); loss,out=m(torch.randn(2,1,5000)); assert loss.ndim==0 and torch.isfinite(loss); loss.backward(); assert out['prediction'].shape[0]==2
 if name!='mae':
  teacher=m.teacher if name=='data2vec' else m.target; assert all(p.grad is None for p in teacher.parameters())
def test_ema_changes_target():
 m=build_method('data2vec'); before=next(m.teacher.parameters()).detach().clone(); next(m.student.parameters()).data.add_(1);m.update_ema();assert not torch.equal(before,next(m.teacher.parameters()))
