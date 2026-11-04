"""Frozen scientific protocol; paths belong to CLI arguments, not this schema."""
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from src.models.neuralgcm_residual.config import FIELDS, FORCING_FIELDS

LEVELS = (1,2,3,5,7,10,20,30,50,70,100,125,150,175,200,225,250,300,350,400,450,500,550,600,650,700,750,775,800,825,850,875,900,925,950,975,1000)
P = LEVELS[8:]

@dataclass(frozen=True)
class Config:
    schema_version: int = 1
    feedback_mode: str = 'baseline'
    arm: str = 'main'
    seed: int = 22
    segment_steps: int = 96
    bptt_steps: int = 24
    truth_prefix_steps: int = 4
    ar_tail_k: int = 20
    max_steps: int = 10000
    checkpoint_every: int = 500
    validation_every: int = 1000
    stats_stride: int = 1
    precision: str = 'fp32'
    learning_rate: float = 1e-4
    end_learning_rate: float = 1e-5
    warmup_steps: int = 200
    adam_beta1: float = .9
    adam_beta2: float = .98
    weight_decay: float = 1e-4
    grad_clip: float = 1.
    mesh_size: int = 4
    width: int = 512
    msg_steps: int = 2
    temporal_d_inner: int = 128
    temporal_layers: int = 2
    temporal_d_state: int = 16
    temporal_d_conv: int = 4
    temporal_bc_groups: int = 1
    temporal_dt_rank: str = 'auto'
    temporal_init_scheme: str = 'mamba1'
    temporal_dt_init: str = 'random'
    temporal_dt_min: float = 1e-3
    temporal_dt_max: float = .1
    temporal_dt_scale: float = 1.
    temporal_dt_init_floor: float = 1e-4
    temporal_conv_bias: bool = True
    temporal_bias: bool = False
    temporal_dropout: float = 0.
    temporal_zero_init_out: bool = True
    temporal_stateful: bool = True
    input_steps: int = 2
    input_channels_main: int = 796
    output_channels: int = 203
    corrected_level_min_hpa: int = 50
    forecast_steps: int = 40
    forcing_policy: str = 'lag24h_then_persist_at_origin'
    loss_mode: str = 'all_steps'
    radius_query_fraction_edge_length: float = .6
    cloud_alpha: float = .05
    cloud_scale_floor: float = 1e-3
    swa_steps: tuple = tuple(range(2000,8001,500))

    def __post_init__(self):
        # Only the predeclared controls and sampling option may vary in v1.
        defaults = {f.name: f.default for f in self.__dataclass_fields__.values()}
        for name, value in asdict(self).items():
            if name not in ('arm', 'stats_stride') and value != defaults[name]:
                raise ValueError(f'{name} changes the v1 protocol; version the specification first')
        if self.arm not in ('main','C2','C3','C4') or self.stats_stride not in (1,7):
            raise ValueError('arm must be main/C2/C3/C4; stats_stride must be 1 or 7')

    @property
    def channels(self):
        return 537 if self.arm == 'C4' else 797 if self.arm == 'C3' else 796

    @property
    def reset_every_step(self):
        return self.arm in ('C2','C3')

    def architecture(self):
        from src.models.mamba.v24_Ilya.config import V24IlyaArchitectureConfig
        return V24IlyaArchitectureConfig(resolution=2.8, mesh_size=self.mesh_size,
            width=self.width, residual_initialization='fresh', residual_msg_steps=self.msg_steps,
            temporal_d_inner=self.temporal_d_inner, temporal_init_scheme='mamba1',
            temporal_zero_init_out=True, temporal_layers=self.temporal_layers,
            temporal_d_state=self.temporal_d_state, temporal_d_conv=self.temporal_d_conv,
            temporal_bc_groups=self.temporal_bc_groups, temporal_dt_rank=self.temporal_dt_rank,
            temporal_dt_init=self.temporal_dt_init, temporal_dt_min=self.temporal_dt_min,
            temporal_dt_max=self.temporal_dt_max, temporal_dt_scale=self.temporal_dt_scale,
            temporal_dt_init_floor=self.temporal_dt_init_floor, temporal_conv_bias=self.temporal_conv_bias,
            temporal_bias=self.temporal_bias, temporal_dropout=self.temporal_dropout,
            temporal_stateful=self.temporal_stateful)

    def v24(self):
        from src.models.mamba.v24_Ilya.training.config import V24IlyaTrainConfig
        return V24IlyaTrainConfig(prepared_root=Path('.'), anchor_manifest_root=Path('.'),
            baseline_checkpoint=Path('.'), output_root=Path('.'), run_name='residualmamba_ngcm',
            architecture=self.architecture(), segment_steps=96, bptt_steps=24, ar_tail_k=20,
            feedback_mode='baseline', loss_mode='all_steps', precision='fp32', seed=22,
            max_steps=self.max_steps, checkpoint_every=500, learning_rate=self.learning_rate,
            end_learning_rate=self.end_learning_rate, learning_rate_schedule='cosine',
            warmup_steps=200, adam_beta1=.9, adam_beta2=.98, weight_decay=1e-4, grad_clip=1.)

    @classmethod
    def load(cls, path):
        value = json.loads(Path(path).read_text())
        if 'swa_steps' in value:
            value['swa_steps'] = tuple(value['swa_steps'])
        return cls(**value)
