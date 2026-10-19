"""Training-consistent spherical diagnostics, independent of the zonal plots.

Each initialization retains physical, unnormalized sufficient statistics for
the nine selected fields and two derived wind speeds by default, or all
primitive variable/level channels with the full-field factory. Degree zero is retained
in the files and excluded explicitly when assessing the anomaly spectrum.
The fitted continuous spherical norm and the discrete grid norm are reported
separately; their difference is not called unresolved energy.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re

import jax
import jax.numpy as jnp
import numpy as np

from .spectral_observer import SPECTRAL_UNITS, SPECTRAL_VARIABLES, extract_fields, spectral_fields
from .training.spectral_loss import _legendre_basis, build_spherical_transform, latitude_area_weights


def diagnostic_kernel(latitude, longitude, lmax=None):
    """Return a compiled analysis and a discrete projection closure check."""
    transform = build_spherical_transform(latitude, longitude, lmax)
    basis = jnp.asarray(_legendre_basis(latitude, transform.lmax), dtype=jnp.float32)
    area = jnp.asarray(latitude_area_weights(latitude), dtype=jnp.float32)
    multiplicity = jnp.where(jnp.arange(transform.lmax + 1) == 0, 1., 2.)

    @jax.jit
    def calculate(fields):
        coefficients = transform.analysis(fields)
        power = jnp.sum(multiplicity * jnp.abs(coefficients) ** 2, axis=-1)
        cross = jnp.sum(multiplicity * jnp.real(coefficients[1:] * coefficients[0].conj()), axis=-1)
        field_error = fields[1:] - fields[0]
        error_coefficients = transform.analysis(field_error)
        error = jnp.sum(multiplicity * jnp.abs(error_coefficients) ** 2, axis=-1)
        # Synthesis of the least-squares projection. Components outside T(lmax)
        # are zero; rfft/irfft scaling is explicitly reversed.
        def synthesis(value):
            real = jnp.einsum('lmj,...lm->...jm', basis, value.real,
                              precision=jax.lax.Precision.HIGHEST)
            imag = jnp.einsum('lmj,...lm->...jm', basis, value.imag,
                              precision=jax.lax.Precision.HIGHEST)
            rings = jax.lax.complex(real, imag)
            rings = jnp.pad(rings, [(0, 0)] * (rings.ndim - 1) +
                           [(0, len(longitude) // 2 + 1 - rings.shape[-1])])
            return jnp.fft.irfft(rings * len(longitude), n=len(longitude), axis=-1)

        projected = synthesis(coefficients)
        # Analyze errors before reconstruction to preserve small forecast
        # differences in fields with a large offset (especially surface pressure).
        projected_error = synthesis(error_coefficients)

        def grid_power(x):
            return jnp.sum(jnp.mean(x * x, axis=-1) * area, axis=-1)

        unresolved = fields - projected
        return {
            'truth_power': power[0], 'baseline_power': power[1], 'full_power': power[2],
            'baseline_cross_power': cross[0], 'full_cross_power': cross[1],
            'baseline_error_power': error[0], 'full_error_power': error[1],
            'truth_unresolved_grid_power': grid_power(unresolved[0]),
            'baseline_unresolved_grid_power': grid_power(unresolved[1]),
            'full_unresolved_grid_power': grid_power(unresolved[2]),
            'baseline_grid_mse': grid_power(field_error[0]),
            'full_grid_mse': grid_power(field_error[1]),
            'baseline_projected_grid_mse': grid_power(projected_error[0]),
            'full_projected_grid_mse': grid_power(projected_error[1]),
            'baseline_unresolved_grid_mse': grid_power(field_error[0] - projected_error[0]),
            'full_unresolved_grid_mse': grid_power(field_error[1] - projected_error[1]),
        }

    return transform, calculate


class SphericalObserver:
    def __init__(self, output_dir, identity, *, target_steps=40, lmax=179,
                 all_channels=False):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.identity = dict(identity)
        self.target_steps = target_steps
        self.lmax = lmax
        self.all_channels = all_channels
        self._variables = tuple(SPECTRAL_VARIABLES)
        self._units = tuple(SPECTRAL_UNITS)
        self._channel_layout = None
        self._metadata = None
        self._grid = None

    def _fields(self, dataset):
        if not self.all_channels:
            return spectral_fields(extract_fields(dataset))
        units_by_variable = {
            '2m_temperature': 'K', 'temperature': 'K',
            '10m_u_component_of_wind': 'm/s', '10m_v_component_of_wind': 'm/s',
            'u_component_of_wind': 'm/s', 'v_component_of_wind': 'm/s',
            'mean_sea_level_pressure': 'Pa', 'total_precipitation_6hr': 'm',
            'specific_humidity': 'kg/kg', 'geopotential': 'm2/s2',
            'vertical_velocity': 'Pa/s',
        }
        layout, fields = [], []
        for name in sorted(dataset.data_vars):
            if name not in units_by_variable:
                raise ValueError(f'Unknown physical spherical variable: {name}')
            variable = dataset[name]
            levels = np.asarray(variable.level).tolist() if 'level' in variable.dims else [None]
            for level in levels:
                field = variable.sel(level=level) if level is not None else variable
                for dim in tuple(field.dims):
                    if dim not in ('lat', 'lon'):
                        if field.sizes[dim] != 1:
                            raise ValueError(f'Spherical observer requires one field per lead: {name}/{dim}')
                        field = field.isel({dim: 0})
                channel = f'{name}_level{level:g}' if level is not None else name
                layout.append((channel, units_by_variable[name]))
                fields.append(np.asarray(field.transpose('lat', 'lon'), dtype=np.float32))
        layout = tuple(layout)
        if not layout:
            raise ValueError('No spherical physical channels')
        if self._channel_layout is None:
            self._channel_layout = layout
            self._variables, self._units = zip(*layout)
        elif layout != self._channel_layout:
            raise ValueError('Spherical physical channel layout changed')
        return np.stack(fields)

    def begin_sample(self, metadata):
        if self._metadata is not None:
            raise ValueError('An initialization is already in progress')
        name = metadata['initialization_id']
        if not re.fullmatch(r'\d{8}T\d{6}Z', name):
            raise ValueError('Invalid initialization ID')
        if metadata['lead_hours'] != list(range(6, 6 * self.target_steps + 1, 6)):
            raise ValueError('Unexpected spherical forecast leads')
        self._destination = self.output_dir / f'{name}.npz'
        if self._destination.exists():
            raise FileExistsError(self._destination)
        self._metadata = {**metadata, **self.identity}
        self._records = []

    def update_step(self, step_index, truth, baseline, full):
        if self._metadata is None or step_index != len(self._records) or step_index >= self.target_steps:
            raise ValueError('Missing or out-of-order spherical forecast step')
        latitude, longitude = np.asarray(truth.lat), np.asarray(truth.lon)
        for value in (baseline, full):
            if not np.array_equal(value.lat, latitude) or not np.array_equal(value.lon, longitude):
                raise ValueError('Spherical forecast and truth grids differ')
        if self._grid is None:
            self._grid = (latitude.copy(), longitude.copy())
            self._transform, self._calculate = diagnostic_kernel(latitude, longitude, self.lmax)
        elif not all(np.array_equal(x, y) for x, y in zip(self._grid, (latitude, longitude))):
            raise ValueError('Spherical grid changed')
        fields = np.stack([self._fields(x) for x in (truth, baseline, full)])
        record = jax.device_get(self._calculate(jnp.asarray(fields, dtype=jnp.float32)))
        if not all(np.isfinite(x).all() for x in record.values()):
            raise ValueError('Nonfinite spherical diagnostics')
        # Orthogonal projection closure uses the discrete area norm, not the
        # continuous sum of spherical coefficient powers.
        for branch in ('baseline', 'full'):
            np.testing.assert_allclose(record[f'{branch}_grid_mse'],
                                       record[f'{branch}_projected_grid_mse'] + record[f'{branch}_unresolved_grid_mse'],
                                       rtol=3e-4, atol=1e-10)
        self._records.append(record)

    def finish_sample(self):
        if self._metadata is None or len(self._records) != self.target_steps:
            raise ValueError('Incomplete spherical initialization')
        metadata = {**self._metadata, 'status': 'complete',
                    'definition': 'Global physical spherical degree powers, area-weighted QR T179; mean degree retained',
                    'lmax': self.lmax, 'normalization': 'physical units; no training variable/level weighting',
                    'field_selection': 'all_primitive_channels' if self.all_channels else 'selected_fields',
                    'grid_mse_definition': 'Exact pole-inclusive discrete latitude areas; separate from continuous spectral norm',
                    'transform_condition_number': self._transform.maximum_condition_number}
        payload = {key: np.stack([x[key] for x in self._records]) for key in self._records[0]}
        payload.update(degree=np.arange(self.lmax + 1), variables=np.asarray(self._variables),
                       units=np.asarray(self._units), lead_hours=np.asarray(self._metadata['lead_hours']),
                       metadata=np.asarray(json.dumps(metadata, sort_keys=True, allow_nan=False)))
        temporary = self._destination.with_suffix('.partial.npz')
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, **payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(self._destination)
        self._metadata = None
        self._records = []


def make_observer(spec, destination, identity):
    return SphericalObserver(Path(destination) / 'spherical', identity,
                             target_steps=spec['target_steps'], lmax=179)


def make_full_field_observer(spec, destination, identity):
    return SphericalObserver(Path(destination) / 'spherical', identity,
                             target_steps=spec['target_steps'], lmax=179,
                             all_channels=True)
