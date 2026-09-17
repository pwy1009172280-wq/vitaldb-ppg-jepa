from .mae import MAE1D
from .data2vec import Data2VecStyle1D
from .jepa import JEPA1D
from .common import BackboneConfig
def build_model(cfg):
    b=BackboneConfig(cfg.model.embed_dim,cfg.model.depth,cfg.model.num_heads,cfg.model.mlp_ratio,cfg.model.dropout,'sinusoidal')
    if cfg.method=='mae': return MAE1D(b,cfg.model.patch_size,cfg.model.patch_stride,cfg.mae.mask_ratio,cfg.model.decoder_dim,cfg.model.decoder_depth,cfg.model.decoder_num_heads,cfg.model.decoder_mlp_ratio)
    if cfg.method=='data2vec': return Data2VecStyle1D(b,cfg.model.patch_size,cfg.model.patch_stride,cfg.data2vec.mask_ratio,cfg.data2vec.mask_span_length,cfg.data2vec.target_top_k_layers,cfg.data2vec.ema_momentum,cfg.data2vec.loss_beta)
    if cfg.method=='jepa': return JEPA1D(b,cfg.model.patch_size,cfg.model.patch_stride,cfg.jepa.num_target_blocks,cfg.jepa.target_block_length,cfg.jepa.predictor_dim,cfg.jepa.predictor_depth,cfg.jepa.predictor_num_heads,cfg.jepa.predictor_mlp_ratio,cfg.jepa.ema_momentum,cfg.jepa.loss_beta)
    raise ValueError(f'unknown method: {cfg.method}')
