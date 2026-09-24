"""Versioned feedback constraints, independent of the cached state schema."""
from .native_state import NativeAdapter, field_value, physical_state, replace

LEGACY_CORRECTION = "native_modal_v1"
CONSTRAINED_CORRECTION = "no_pressure_zero_mean_v2"
CORRECTION_POLICIES = (LEGACY_CORRECTION, CONSTRAINED_CORRECTION)


class ConstrainedNativeAdapter(NativeAdapter):
    """Preserve pressure and pre-existing degree-zero divergence/vorticity.

    The native feature layout and baseline cache are unchanged. The correction
    policy is separately included in run configuration/checkpoint identities.
    Constraints act only on the added increment, never on the baseline state.
    """

    def apply_increment(self, state, increment):
        # Surface pressure remains the prognostic solver's responsibility.
        increment = increment.at[..., -1].set(0)
        corrected = super().apply_increment(state, increment)
        old, core = physical_state(state), physical_state(corrected)
        core = replace(core,
            log_surface_pressure=old.log_surface_pressure,
            divergence=core.divergence.at[..., 0].set(old.divergence[..., 0]),
            vorticity=core.vorticity.at[..., 0].set(old.vorticity[..., 0]))
        return replace(corrected, state=core) if hasattr(corrected, "state") else core


def make_adapter(model, state, policy):
    if policy == LEGACY_CORRECTION:
        return NativeAdapter(model, state)
    if policy == CONSTRAINED_CORRECTION:
        return ConstrainedNativeAdapter(model, state)
    raise ValueError(f"Unknown native correction policy: {policy}")
