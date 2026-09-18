"""Canopy, shade, and glacier-ice adjustments for inversion snow fraction."""

from __future__ import annotations

import numpy as np
import xarray as xr

from spires_postprocess._xarray_validation import (
    prepare_aligned_layer,
    require_float32,
    require_matching_coords,
    require_values_in_range,
    validate_target_layout,
)


_OBSCURATION_LIMIT = 0.99
_GEOMETRY_EPSILON = 1.0e-6
CANOPY_CORRECTION_POLICIES = ("legacy", "modis_s2_062", "viirs_s2_083")


def calculate_viewable_canopy_fraction(
    canopy_fraction: xr.DataArray,
    sensor_zenith: xr.DataArray,
    *,
    sensor_azimuth: xr.DataArray | None = None,
    slope: xr.DataArray | None = None,
    aspect: xr.DataArray | None = None,
    average_vertical_crown_radius: float = 4.644,
    average_horizontal_crown_radius: float = 1.72,
    canopy_source: str | None = None,
) -> xr.DataArray:
    """Return canopy obstruction adjusted for sensor and optional terrain geometry.

    Static ``(y, x)`` canopy and terrain layers broadcast over sensor geometry with
    ``(time, y, x)`` dimensions. Slope and aspect must either both be supplied or
    both be omitted. Terrain-aware calculations expect sensor azimuth and aspect
    in degrees clockwise from north. All scientific input arrays must already be
    float32. NaNs are preserved; invalid finite geometry or fraction values raise.
    """
    _validate_crown_radii(
        average_vertical_crown_radius,
        average_horizontal_crown_radius,
    )
    target_dims = validate_target_layout(sensor_zenith, "sensor_zenith")
    sensor_zenith = _prepare_geometry_layer(
        sensor_zenith,
        sensor_zenith,
        name="sensor_zenith",
        minimum=0.0,
        maximum=90.0,
    )
    canopy = _prepare_fraction_layer(
        canopy_fraction,
        sensor_zenith,
        name="canopy_fraction",
    )

    if (slope is None) != (aspect is None):
        raise ValueError(
            "slope and aspect must either both be provided or both be omitted"
        )

    crown_ratio = average_vertical_crown_radius / average_horizontal_crown_radius
    theta_v_prime = np.arctan(
        crown_ratio * np.tan(np.deg2rad(sensor_zenith))
    )

    terrain_geometry_used = slope is not None
    if terrain_geometry_used:
        if sensor_azimuth is None:
            raise ValueError(
                "sensor_azimuth is required when slope and aspect are provided"
            )
        slope_layer = _prepare_geometry_layer(
            slope,
            sensor_zenith,
            name="slope",
            minimum=0.0,
            maximum=90.0,
        )
        aspect_layer = _prepare_geometry_layer(
            aspect,
            sensor_zenith,
            name="aspect",
            minimum=0.0,
            maximum=360.0,
        )
        normalized_sensor_azimuth = xr.where(
            np.isfinite(sensor_azimuth),
            sensor_azimuth % np.float32(360.0),
            np.nan,
        ).astype("float32")
        sensor_azimuth_layer = _prepare_geometry_layer(
            normalized_sensor_azimuth,
            sensor_zenith,
            name="sensor_azimuth",
            minimum=0.0,
            maximum=360.0,
        )

        slope_rad = np.deg2rad(slope_layer)
        theta_s_prime = np.arctan2(
            np.sin(slope_rad),
            crown_ratio * np.cos(slope_rad),
        )
        phi_v_prime = np.deg2rad(sensor_azimuth_layer - aspect_layer)
        denominator = (
            np.cos(phi_v_prime)
            * np.sin(theta_v_prime)
            * np.sin(theta_s_prime)
            + np.cos(theta_v_prime) * np.cos(theta_s_prime)
        )
        exponent = xr.where(
            denominator > _GEOMETRY_EPSILON,
            np.cos(theta_s_prime) / denominator,
            np.nan,
        )
        model = "liu_2004_go_vgf_terrain"
    else:
        exponent = 1.0 / np.cos(theta_v_prime)
        model = "liu_2004_go_vgf_flat_terrain"

    adjusted = (1.0 - ((1.0 - canopy) ** exponent)).clip(min=0.0, max=1.0)
    adjusted = adjusted.transpose(*target_dims)
    adjusted = adjusted.astype("float32").rename("viewable_canopy_fraction")
    adjusted.attrs = {
        "long_name": "View-angle-adjusted canopy obstruction fraction",
        "units": "1",
        "model": model,
        "source": _source_label(canopy_fraction, canopy_source, "canopy_fraction"),
        "average_vertical_crown_radius": average_vertical_crown_radius,
        "average_horizontal_crown_radius": average_horizontal_crown_radius,
        "terrain_geometry_used": int(terrain_geometry_used),
        "azimuth_convention": "degrees clockwise from north",
        "sensor_azimuth_normalization": "finite values modulo 360",
        "supported_result_dims": ",".join(target_dims),
    }
    return adjusted


def apply_snow_fraction_adjustments(
    results: xr.Dataset,
    *,
    canopy_fraction: xr.DataArray | None = None,
    ice_fraction: xr.DataArray | None = None,
    sensor_zenith: xr.DataArray | None = None,
    sensor_azimuth: xr.DataArray | None = None,
    slope: xr.DataArray | None = None,
    aspect: xr.DataArray | None = None,
    average_vertical_crown_radius: float = 4.644,
    average_horizontal_crown_radius: float = 1.72,
    canopy_source: str | None = None,
    ice_source: str | None = None,
    canopy_correction_policy: str = "legacy",
) -> xr.Dataset:
    """Add requested canopy- and ice-adjusted snow-fraction layers.

    Supplying ``canopy_fraction`` requests ``canopy_adjusted_fsnow``. Supplying
    ``ice_fraction`` requests ``ice_adjusted_fsnow``. When both are supplied, the
    ice-adjusted layer is calculated directly from the original ``fsnow`` using
    shade, viewable canopy, and ice together, matching the SPIRES 2025.0.1 daily
    calculation. The input dataset and its original ``fsnow``/``fshade`` variables
    are not modified. Required results and ancillary inputs must already be
    float32 and within their documented physical ranges.
    """
    if canopy_correction_policy not in CANOPY_CORRECTION_POLICIES:
        raise ValueError(f"unknown canopy_correction_policy: {canopy_correction_policy!r}")
    if canopy_correction_policy != "legacy":
        if canopy_fraction is None or slope is None or aspect is None or sensor_azimuth is None:
            raise ValueError("sensor-specific canopy correction requires canopy, slope, aspect, and sensor azimuth")
        if average_vertical_crown_radius != 4.644 or average_horizontal_crown_radius != 1.72:
            raise ValueError("selected canopy policies require the evaluated crown radii 4.644 and 1.72")
    if canopy_fraction is None and ice_fraction is None:
        raise ValueError("provide canopy_fraction, ice_fraction, or both")

    fsnow, fshade = _validated_results(results)
    adjusted_results = results.copy()
    viewable_canopy = None

    if canopy_fraction is not None:
        if sensor_zenith is None:
            raise ValueError("sensor_zenith is required for canopy correction")
        viewable_canopy = calculate_viewable_canopy_fraction(
            canopy_fraction,
            sensor_zenith,
            sensor_azimuth=sensor_azimuth,
            slope=slope,
            aspect=aspect,
            average_vertical_crown_radius=average_vertical_crown_radius,
            average_horizontal_crown_radius=average_horizontal_crown_radius,
            canopy_source=canopy_source,
        )
        viewable_canopy_attrs = dict(viewable_canopy.attrs)
        viewable_canopy = _prepare_fraction_layer(
            viewable_canopy,
            fsnow,
            name="viewable_canopy_fraction",
        )
        viewable_canopy.attrs = viewable_canopy_attrs
        canopy_obscuration = (fshade + viewable_canopy).clip(
            min=0.0,
            max=_OBSCURATION_LIMIT,
        )
        canopy_adjusted = (fsnow / (1.0 - canopy_obscuration)).clip(
            min=0.0,
            max=1.0,
        )
        canopy_adjusted = canopy_adjusted.astype("float32").rename(
            "canopy_adjusted_fsnow"
        )
        canopy_adjusted.attrs = {
            "long_name": "Snow fraction adjusted for shade and canopy obstruction",
            "units": "1",
            "model": "spires_2025_0_1_obscuration",
            "canopy_correction_applied": 1,
            "shade_included": 1,
            "canopy_included": 1,
            "ice_included": 0,
            "canopy_source": viewable_canopy.attrs.get("source", "canopy_fraction"),
            "average_vertical_crown_radius": average_vertical_crown_radius,
            "average_horizontal_crown_radius": average_horizontal_crown_radius,
            "terrain_geometry_used": viewable_canopy.attrs.get(
                "terrain_geometry_used", 0
            ),
            "obscuration_upper_bound": _OBSCURATION_LIMIT,
        }
        adjusted_results["canopy_adjusted_fsnow"] = canopy_adjusted
        if canopy_correction_policy != "legacy":
            canopy = _prepare_fraction_layer(canopy_fraction, fsnow, name="canopy_fraction")
            is_modis = canopy_correction_policy == "modis_s2_062"
            rho = np.float32(0.1 if is_modis else 0.5)
            weight = xr.ones_like(canopy) if is_modis else (
                np.float32(1) - np.float32(0.75)
                * ((canopy - np.float32(0.05)) / np.float32(0.25)).clip(0, 1)
            )
            shade = weight * fshade
            full_denominator = np.float32(1) - (shade + viewable_canopy).clip(0, np.float32(0.99))
            reference_denominator = np.float32(1) - (shade + canopy).clip(0, np.float32(0.99))
            full_gain = np.float32(1) / full_denominator
            reference_gain = np.float32(1) / reference_denominator
            geometry_gain = full_gain / reference_gain
            amplification = geometry_gain > np.float32(1)
            gain = xr.where(
                amplification,
                reference_gain * np.power(geometry_gain, xr.full_like(geometry_gain, rho)),
                full_gain,
            )
            # Preserve the frozen experiment's branch arithmetic in float32.
            preclip = xr.where(amplification, fsnow * gain, fsnow / full_denominator)
            preclip = xr.where(gain > np.float32(5), fsnow * np.float32(5), preclip)
            corrected = preclip.clip(0, 1).astype("float32")
            corrected.attrs = dict(canopy_adjusted.attrs)
            corrected.attrs.update(
                model="spires_sensor_specific_canopy_20260918",
                canopy_correction_policy=canopy_correction_policy,
                candidate_id="S2_062" if is_modis else "S2_083",
                geometry_gain_exponent=float(rho),
                geometry_regularization="amplification_only",
                shade_treatment="full" if is_modis else "ramp_1_at_0.05_to_0.25_at_0.30",
                total_gain_cap=5.0,
                detection_applied=0,
            )
            adjusted_results["canopy_adjusted_fsnow"] = corrected
            threshold = np.float32(0.05 if is_modis else 0.04)
            detected = xr.where(fsnow.notnull(), (fsnow > threshold).astype("float32"), np.float32(np.nan))
            detected.attrs = {
                "long_name": "Raw snow detection, independent of canopy-adjusted magnitude and QA eligibility",
                "units": "1", "raw_fsnow_threshold": float(threshold),
                "comparison": "strictly_greater_than", "qa_eligibility_applied": 0,
                "canopy_correction_policy": canopy_correction_policy,
            }
            adjusted_results["snow_detected"] = detected.astype("float32")

    if ice_fraction is not None:
        ice = _prepare_fraction_layer(ice_fraction, fsnow, name="ice_fraction")
        total_obscuration = fshade + ice
        if viewable_canopy is not None:
            total_obscuration = total_obscuration + viewable_canopy
        total_obscuration = total_obscuration.clip(
            min=0.0,
            max=_OBSCURATION_LIMIT,
        )
        ice_adjusted = (fsnow / (1.0 - total_obscuration)).clip(
            min=0.0,
            max=1.0,
        )
        ice_adjusted = xr.where(ice_adjusted < ice, ice, ice_adjusted)
        ice_adjusted = ice_adjusted.astype("float32").rename("ice_adjusted_fsnow")
        ice_adjusted_attrs = {
            "long_name": "Snow fraction adjusted for shade, canopy, and glacier ice",
            "units": "1",
            "model": "spires_2025_0_1_obscuration_and_ice_floor",
            "ice_adjustment_applied": 1,
            "shade_included": 1,
            "canopy_included": int(viewable_canopy is not None),
            "ice_included": 1,
            "ice_source": _source_label(ice_fraction, ice_source, "ice_fraction"),
            "canopy_source": (
                viewable_canopy.attrs.get("source", "canopy_fraction")
                if viewable_canopy is not None
                else "none"
            ),
            "terrain_geometry_used": (
                viewable_canopy.attrs.get("terrain_geometry_used", 0)
                if viewable_canopy is not None
                else 0
            ),
            "obscuration_upper_bound": _OBSCURATION_LIMIT,
            "ice_lower_bound_applied": 1,
        }
        if viewable_canopy is not None:
            ice_adjusted_attrs.update(
                average_vertical_crown_radius=average_vertical_crown_radius,
                average_horizontal_crown_radius=average_horizontal_crown_radius,
            )
        ice_adjusted.attrs = ice_adjusted_attrs
        adjusted_results["ice_adjusted_fsnow"] = ice_adjusted

    return adjusted_results


def _validated_results(results: xr.Dataset) -> tuple[xr.DataArray, xr.DataArray]:
    if not isinstance(results, xr.Dataset):
        raise TypeError("results must be an xarray.Dataset")
    missing = [name for name in ("fsnow", "fshade") if name not in results]
    if missing:
        raise ValueError(f"results is missing required variable(s): {missing}")

    fsnow = results["fsnow"]
    fshade = results["fshade"]
    validate_target_layout(fsnow, "results['fsnow']")
    require_float32(fsnow, "results['fsnow']")
    require_float32(fshade, "results['fshade']")
    require_values_in_range(
        fsnow,
        "results['fsnow']",
        minimum=0.0,
        maximum=1.0,
    )
    require_values_in_range(
        fshade,
        "results['fshade']",
        minimum=0.0,
        maximum=1.0,
    )
    if fshade.dims != fsnow.dims:
        raise ValueError(
            "results['fshade'] must have the same dimensions as results['fsnow']"
        )
    require_matching_coords(
        fshade,
        fsnow,
        fsnow.dims,
        label="results['fshade']",
        target_label="results['fsnow']",
    )
    return fsnow, fshade


def _prepare_fraction_layer(
    layer: xr.DataArray,
    target: xr.DataArray,
    *,
    name: str,
) -> xr.DataArray:
    prepared = prepare_aligned_layer(layer, target, label=name)
    require_float32(prepared, name)
    require_values_in_range(prepared, name, minimum=0.0, maximum=1.0)
    return prepared


def _prepare_geometry_layer(
    layer: xr.DataArray,
    target: xr.DataArray,
    *,
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> xr.DataArray:
    prepared = prepare_aligned_layer(layer, target, label=name)
    require_float32(prepared, name)
    require_values_in_range(
        prepared,
        name,
        minimum=minimum,
        maximum=maximum,
    )
    return prepared


def _validate_crown_radii(vertical: float, horizontal: float) -> None:
    if vertical <= 0:
        raise ValueError("average_vertical_crown_radius must be > 0")
    if horizontal <= 0:
        raise ValueError("average_horizontal_crown_radius must be > 0")


def _source_label(
    layer: xr.DataArray,
    explicit_source: str | None,
    fallback: str,
) -> str:
    if explicit_source is not None:
        return str(explicit_source)
    return str(layer.attrs.get("source", fallback))
