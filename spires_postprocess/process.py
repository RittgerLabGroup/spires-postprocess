"""High-level configurable postprocessing for shared ``SpiresData`` objects."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from spires_contract import (
    SpiresData,
    validate_results,
    validate_spires_data,
)
import xarray as xr

from spires_postprocess.lookup import (
    ALTITUDE,
    SKYVIEW,
    load_albedo_lookup,
    load_forcing_lookup,
)
from spires_postprocess.snow_fraction import (
    CANOPY_CORRECTION_POLICIES,
    apply_snow_fraction_adjustments,
)
from spires_postprocess.snow_radiative import (
    compute_delta_vis,
    compute_radiative_forcing,
    compute_snow_albedo,
)


_BASE_RESULT_NAMES = (
    "fsnow",
    "fshade",
    "lap_concentration",
    "grain_radius",
)
_ALBEDO_RESULT_NAMES = (
    "albedo_clean_flat",
    "albedo_dirty_flat",
    "albedo_clean_terrain_corrected",
    "albedo_dirty_terrain_corrected",
)
_METRE_UNITS = {"m", "meter", "meters", "metre", "metres"}
_KILOMETRE_UNITS = {"km", "kilometer", "kilometers", "kilometre", "kilometres"}


def process(
    data: SpiresData,
    *,
    apply_canopy_correction: bool = False,
    apply_ice_adjustment: bool = False,
    calculate_albedo: bool = False,
    calculate_delta_vis: bool = False,
    calculate_radiative_forcing: bool = False,
    albedo_lookup: xr.Dataset | str | Path | None = None,
    forcing_lookup: xr.Dataset | str | Path | None = None,
    average_vertical_crown_radius: float = 4.644,
    average_horizontal_crown_radius: float = 1.72,
    canopy_correction_policy: str = "legacy",
) -> SpiresData:
    """Apply explicitly selected postprocessing operations.

    The Boolean options intentionally match the main run configuration's
    ``postprocess`` section. This package does not import that configuration
    type; callers translate its values into these explicit keyword arguments.
    Scene and ancillary inputs use strict canonical field names, and a requested
    operation fails with an actionable error when one is absent.
    """
    options = {
        "apply_canopy_correction": apply_canopy_correction,
        "apply_ice_adjustment": apply_ice_adjustment,
        "calculate_albedo": calculate_albedo,
        "calculate_delta_vis": calculate_delta_vis,
        "calculate_radiative_forcing": calculate_radiative_forcing,
    }
    for name, value in options.items():
        if type(value) is not bool:
            raise TypeError(f"{name} must be a boolean")
    if canopy_correction_policy not in CANOPY_CORRECTION_POLICIES:
        raise ValueError(f"unknown canopy_correction_policy: {canopy_correction_policy!r}")
    if canopy_correction_policy != "legacy" and not apply_canopy_correction:
        raise ValueError("sensor-specific canopy policy requires apply_canopy_correction=True")

    validate_spires_data(data)
    if data.results is None:
        raise ValueError("data.results is required for postprocessing")
    validate_results(data.results, scene=data.scene)

    if not any(options.values()):
        return data

    original_results = data.results
    updated = original_results.copy(deep=False)

    if apply_canopy_correction or apply_ice_adjustment:
        ancillary = _require_ancillary(data)
        canopy_fraction = (
            _require_variable(
                ancillary,
                "canopy_fraction",
                owner="data.ancillary",
            )
            if apply_canopy_correction
            else None
        )
        ice_fraction = (
            _require_variable(
                ancillary,
                "ice_fraction",
                owner="data.ancillary",
            )
            if apply_ice_adjustment
            else None
        )

        sensor_zenith = None
        sensor_azimuth = None
        slope = None
        aspect = None
        if apply_canopy_correction:
            sensor_zenith = _require_variable(
                data.scene,
                "sensor_zenith",
                owner="data.scene",
            )
            slope = ancillary.get("slope")
            aspect = ancillary.get("aspect")
            if (slope is None) != (aspect is None):
                raise ValueError(
                    "data.ancillary must contain both 'slope' and 'aspect', "
                    "or neither, for canopy correction"
                )
            if slope is not None:
                sensor_azimuth = _require_variable(
                    data.scene,
                    "sensor_azimuth",
                    owner="data.scene",
                )

        updated = apply_snow_fraction_adjustments(
            updated,
            canopy_fraction=canopy_fraction,
            ice_fraction=ice_fraction,
            sensor_zenith=sensor_zenith,
            sensor_azimuth=sensor_azimuth,
            slope=slope,
            aspect=aspect,
            average_vertical_crown_radius=average_vertical_crown_radius,
            average_horizontal_crown_radius=average_horizontal_crown_radius,
            canopy_correction_policy=canopy_correction_policy,
        )

    if calculate_albedo:
        if albedo_lookup is None:
            raise ValueError(
                "albedo_lookup is required when calculate_albedo=True"
            )
        lookup = load_albedo_lookup(albedo_lookup)
        skyview = None
        skyview_source = "not_used"
        skyview_default_applied = False
        if SKYVIEW in lookup["albedo"].dims:
            supplied_skyview = (
                None
                if data.ancillary is None
                else data.ancillary.get(SKYVIEW)
            )
            if supplied_skyview is None:
                skyview = xr.ones_like(
                    updated["grain_radius"],
                    dtype=np.float32,
                ).rename(SKYVIEW)
                skyview.attrs = {
                    "long_name": "Default unobstructed sky-view fraction",
                    "units": "1",
                    "source": "spires_postprocess_default_open_sky",
                }
                skyview_source = "default_open_sky"
                skyview_default_applied = True
            else:
                skyview = supplied_skyview
                skyview_source = "data.ancillary.skyview"
        altitude = (
            _altitude_km(
                _require_variable(
                    _require_ancillary(data),
                    "dem",
                    owner="data.ancillary",
                )
            )
            if ALTITUDE in lookup["albedo"].dims
            else None
        )
        updated = compute_snow_albedo(
            updated,
            cosine_solar_zenith=_require_variable(
                data.scene,
                "cosine_solar_zenith",
                owner="data.scene",
            ),
            cosine_illumination=_require_variable(
                data.scene,
                "cosine_illumination",
                owner="data.scene",
            ),
            lookup=lookup,
            skyview=skyview,
            altitude=altitude,
        )
        if SKYVIEW in lookup["albedo"].dims:
            for name in _ALBEDO_RESULT_NAMES:
                attrs = dict(updated[name].attrs)
                attrs.update(
                    skyview_source=skyview_source,
                    skyview_default_applied=int(skyview_default_applied),
                )
                if skyview_default_applied:
                    attrs["skyview_default_value"] = 1.0
                updated[name].attrs = attrs

    if calculate_delta_vis or calculate_radiative_forcing:
        if forcing_lookup is None:
            raise ValueError(
                "forcing_lookup is required when delta-VIS or radiative "
                "forcing is requested"
            )
        lookup = load_forcing_lookup(forcing_lookup)
        cosine_solar_zenith = _require_variable(
            data.scene,
            "cosine_solar_zenith",
            owner="data.scene",
        )
        if calculate_delta_vis:
            updated = compute_delta_vis(
                updated,
                cosine_solar_zenith=cosine_solar_zenith,
                lookup=lookup,
            )
        if calculate_radiative_forcing:
            updated = compute_radiative_forcing(
                updated,
                cosine_solar_zenith=cosine_solar_zenith,
                lookup=lookup,
            )

    updated = updated.assign_coords(
        {name: coordinate for name, coordinate in original_results.coords.items()}
    )
    _require_unchanged_base_results(original_results, updated)
    validate_results(updated, scene=data.scene)
    return data.assign_results(updated)


def _require_ancillary(data: SpiresData) -> xr.Dataset:
    if data.ancillary is None:
        raise ValueError("data.ancillary is required by the selected postprocessing")
    return data.ancillary


def _require_variable(
    dataset: xr.Dataset,
    name: str,
    *,
    owner: str,
) -> xr.DataArray:
    if name not in dataset:
        raise ValueError(f"{owner} is missing required variable {name!r}")
    return dataset[name]


def _altitude_km(dem: xr.DataArray) -> xr.DataArray:
    units = dem.attrs.get("units")
    if not isinstance(units, str):
        raise ValueError(
            "data.ancillary['dem'] must declare metre or kilometre units "
            "when the albedo lookup includes altitude"
        )
    normalized = units.strip().casefold()
    if normalized in _METRE_UNITS:
        altitude = (dem / np.float32(1000.0)).astype("float32")
        source_units = units
    elif normalized in _KILOMETRE_UNITS:
        altitude = dem.astype("float32")
        source_units = units
    else:
        raise ValueError(
            "data.ancillary['dem'] units must identify metres or kilometres; "
            f"got {units!r}"
        )
    altitude = altitude.rename("altitude")
    altitude.attrs = {
        "long_name": "Physical altitude",
        "units": "km",
        "source_variable": "dem",
        "source_units": source_units,
    }
    return altitude


def _require_unchanged_base_results(
    original: xr.Dataset,
    updated: xr.Dataset,
) -> None:
    for name in _BASE_RESULT_NAMES:
        if not original[name].identical(updated[name]):
            raise RuntimeError(
                f"postprocessing unexpectedly modified base result {name!r}"
            )
