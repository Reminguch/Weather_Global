"""Versioned, explicit contracts for the complete eight-arm experiment."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from . import SCHEMA
from .io import digest, read_json

FIELDS = ("temperature", "geopotential", "u_component_of_wind", "v_component_of_wind",
          "specific_humidity", "specific_cloud_ice_water_content", "specific_cloud_liquid_water_content")
FORCING_FIELDS = ("sea_surface_temperature", "sea_ice_cover")
STAGES = ("preflight", "data", "cache", "verify-cache", "pretrain", "finetune", "evaluate", "report")
FORCING_POLICY = "lag24h_then_persist_at_origin"
LOSS_NAME = "neuralgcm_field_normalized_mse_v1"
BALANCED_LOSS_NAME = "neuralgcm_pooled_change_mse_v2"
CORRECTION_POLICIES = ("native_modal_v1", "no_pressure_zero_mean_v2")


@dataclass(frozen=True)
class Architecture:
    width: int = 128
    d_inner: int = 16
    mesh_size: int = 4
    msg_steps: int = 2
    temporal_layers: int = 2
    d_state: int = 16
    d_conv: int = 4
    bc_groups: int = 1
    dt_rank: int = 32
    init_scheme: str = "mamba1"
    dropout: float = 0.0
    initialization: str = "fresh_zero_head"
    precision: str = "fp32"

    def __post_init__(self):
        if self.width not in (128, 256) or self.d_inner not in (16, 32):
            raise ValueError("Initial matrix requires widths 128/256 and d_inner 16/32")
        if (self.mesh_size, self.msg_steps, self.temporal_layers, self.d_state,
            self.d_conv, self.bc_groups, self.dt_rank) != (4, 2, 2, 16, 4, 1, 32):
            raise ValueError("Fixed architecture defaults changed; version the experiment")
        if (self.init_scheme, self.dropout, self.initialization, self.precision) != (
                "mamba1", 0.0, "fresh_zero_head", "fp32"):
            raise ValueError("Initial experiment requires fresh FP32 Mamba1 with a zero head")


@dataclass(frozen=True)
class Optimizer:
    learning_rate: float
    warmup_updates: int
    beta1: float = 0.9
    beta2: float = 0.98
    weight_decay: float = 1e-4
    clip_norm: float = 1.0


@dataclass(frozen=True)
class RunConfig:
    resolution: float = 2.8
    architecture: Architecture = field(default_factory=Architecture)
    schema: str = SCHEMA
    seed: int = 22
    train_years: tuple[int, ...] = (2015, 2016, 2017, 2018, 2019, 2020, 2021)
    validation_year: int = 2022
    test_year: int = 2023
    step_hours: int = 6
    bptt_steps: int = 24
    segment_steps: int = 96
    pretrain_epochs: int = 20
    rollout_steps: int = 20
    finetune_updates: int = 2000
    finetune_validate_every: int = 200
    validation_origins: int = 32
    test_origins: int = 128
    test_steps: int = 40
    pretrain_optimizer: Optimizer = field(default_factory=lambda: Optimizer(1e-4, 200))
    finetune_optimizer: Optimizer = field(default_factory=lambda: Optimizer(1e-5, 100))
    forcing_policy: str = FORCING_POLICY
    feedback_mode: str = "closed_loop_sg"
    loss: str = LOSS_NAME
    correction_policy: str = "native_modal_v1"

    def __post_init__(self):
        if self.schema != SCHEMA or self.resolution not in (2.8, 1.4):
            raise ValueError("Unsupported schema or resolution")
        if tuple(self.train_years) != tuple(range(2015, 2022)) or (self.validation_year, self.test_year) != (2022, 2023):
            raise ValueError("User-selected split is 2015–2021 train / 2022 validation / 2023 test")
        expected = (22, 6, 24, 96, 20, 20, 2000, 200, 32, 128, 40)
        actual = (self.seed, self.step_hours, self.bptt_steps, self.segment_steps,
                  self.pretrain_epochs, self.rollout_steps, self.finetune_updates,
                  self.finetune_validate_every, self.validation_origins, self.test_origins, self.test_steps)
        if actual != expected:
            raise ValueError("Initial experiment budgets/horizons are locked; use a new schema for changes")
        if ((self.forcing_policy, self.feedback_mode) != (FORCING_POLICY, "closed_loop_sg")
                or self.loss not in (LOSS_NAME, BALANCED_LOSS_NAME)):
            raise ValueError("Unsupported causal/gradient/objective contract")
        if self.correction_policy not in CORRECTION_POLICIES:
            raise ValueError("Unsupported native correction contract")

    @property
    def resolution_id(self):
        return "res" + str(self.resolution).replace(".", "p")

    @property
    def run_id(self):
        return f"r{str(self.resolution).replace('.', 'p')}_w{self.architecture.width}_di{self.architecture.d_inner}"

    def to_dict(self):
        result = asdict(self)
        # Keep the identities of historical configuration files unchanged.
        if self.correction_policy == "native_modal_v1":
            result.pop("correction_policy")
        return result

    @property
    def identity(self):
        return digest(self.to_dict())


def load_config(path):
    value = read_json(path)
    value["architecture"] = Architecture(**value["architecture"])
    value["train_years"] = tuple(value["train_years"])
    for name in ("pretrain_optimizer", "finetune_optimizer"):
        value[name] = Optimizer(**value[name])
    return RunConfig(**value)


def initial_matrix():
    return [RunConfig(resolution=r, architecture=Architecture(width=w, d_inner=d))
            for r in (2.8, 1.4) for w in (128, 256) for d in (16, 32)]
