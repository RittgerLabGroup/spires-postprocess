# spires-postprocess

Platform-agnostic scientific postprocessing for the
[SPIReS](https://github.com/SPIReS-Organization) package family. The package
consumes the canonical inversion boundary defined by
[`spires-contract`](https://github.com/SPIReS-Organization/spires-contract):

- `fsnow`
- `fshade`
- `lap_concentration`
- `grain_radius`

Legacy result names are not accepted or emitted.

## Configurable `SpiresData` workflow

`process()` is a convenience wrapper around the independent xarray-first
operations. Its explicit Boolean options match the main run configuration's
`postprocess` section, but this package does not import configuration classes
from `spires-io`:

```python
from spires_postprocess import process

options = run_config.postprocess
processed = process(
    inverted,
    apply_canopy_correction=options.apply_canopy_correction,
    sensor="viirs",
    canopy_correction_policy=(
        "viirs_canopy_weighted_shade" if options.apply_canopy_correction else None
    ),
    apply_ice_adjustment=options.apply_ice_adjustment,
    calculate_albedo=options.calculate_albedo,
    calculate_delta_vis=options.calculate_delta_vis,
    calculate_radiative_forcing=options.calculate_radiative_forcing,
    average_vertical_crown_radius=options.average_vertical_crown_radius,
    average_horizontal_crown_radius=options.average_horizontal_crown_radius,
    albedo_lookup=albedo_lut_path,
    forcing_lookup=forcing_lut_path,
)
```

All operations default to disabled. A requested operation uses strict field
names and raises an actionable error when an input is absent. The wrapper
replaces only `data.results`; it preserves the scene, background, ancillary
data, and the four base inversion results.

Required object fields are:

| Operation | Required fields |
| --- | --- |
| Canopy correction | Explicit matching `sensor` and `canopy_correction_policy`; `results.fsnow`, `results.fshade`, `ancillary.canopy_fraction`, `scene.sensor_zenith`, `ancillary.slope`, `ancillary.aspect`, and `scene.sensor_azimuth` |
| Ice adjustment | `results.fsnow`, `results.fshade`, `ancillary.ice_fraction` |
| Snow albedo | `results.grain_radius`, `results.lap_concentration`, `scene.cosine_solar_zenith`, `scene.cosine_illumination`, and a canonical albedo LUT |
| Delta-VIS / radiative forcing | `results.grain_radius`, `results.lap_concentration`, `scene.cosine_solar_zenith`, and a canonical DV/RF LUT |

For an albedo LUT with a `skyview` axis, the wrapper uses
`ancillary.skyview` when supplied. If it is absent, the wrapper explicitly
defaults to `skyview=1.0`, representing unobstructed open sky, and records that
fallback in the albedo product metadata. An `altitude` axis still requires
`ancillary.dem` with explicit metre or kilometre units; the wrapper converts
declared metre values to kilometres.

## Canopy and ice adjustments

Both `process()` and `apply_snow_fraction_adjustments()` require an explicit
sensor and matching policy when canopy correction is requested:

| Sensor | `canopy_correction_policy` | Shade treatment |
| --- | --- | --- |
| `modis` | `modis_full_shade` | Full shade |
| `viirs` | `viirs_canopy_weighted_shade` | Canopy-dependent shade weighting |

Missing policies, unknown sensors, cross-sensor policies, `legacy`, and retired
research identifiers are rejected. There is no implicit legacy canopy algorithm.
Omit the policy when canopy correction is disabled; ice-only and radiative-only
operations do not require a canopy policy or sensor. The example above is for
VIIRS; MODIS callers must supply the MODIS sensor and policy together.

Both selected policies require terrain-aware projection with crown radii
4.644/1.72. Let C be canopy fraction, V projected canopy, and S weighted shade.
MODIS uses full shade and exponent rho=0.1. VIIRS uses a shade weight decreasing
linearly from 1 at C=0.05 to 0.25 at C=0.30, constant outside that interval,
and rho=0.5. Set GF=1/(1-clip(S+V,0,0.99)), GR=1/(1-clip(S+C,0,0.99)),
and H=GF/GR. Use GR*H**rho only where H>1, otherwise GF. Cap total gain at 5
and corrected snow fraction at 1. The descriptive policy, sensor and parameters are
persisted on `canopy_adjusted_fsnow`.

Raw `fsnow` and `fshade` remain intact. The independent float32 `snow_detected`
layer uses raw fsnow >0.05 for MODIS and >0.04 for VIIRS, retaining missing
raw observations as NaN. Detection does not zero the corrected magnitude;
cloud/quality eligibility remains a separate downstream decision. The existing
`ice_adjusted_fsnow` equation remains a separate legacy ice product; the new
canopy policy does not change it. Renaming the policies does not change their
evaluated numerical behavior. Update direct callers and regenerate batch plans;
do not edit immutable historical manifests to substitute new labels.

The lower-level API remains independently usable:

```python
from spires_postprocess import apply_snow_fraction_adjustments

adjusted = apply_snow_fraction_adjustments(
    inversion_results,
    sensor="viirs",
    canopy_correction_policy="viirs_canopy_weighted_shade",
    canopy_fraction=data.ancillary["canopy_fraction"],
    ice_fraction=data.ancillary["ice_fraction"],
    sensor_zenith=data.scene["sensor_zenith"],
    sensor_azimuth=data.scene["sensor_azimuth"],
    slope=data.ancillary["slope"],
    aspect=data.ancillary["aspect"],
)
```

Supplying canopy data adds `canopy_adjusted_fsnow`. Supplying ice data adds
`ice_adjusted_fsnow`. Both use the original `fsnow`. Canopy correction uses the
sensor-specific gain described above; the independent ice equation remains:

```text
O_ice = clip(fshade + viewable_canopy_fraction + ice_fraction, 0, 0.99)
ice_adjusted_fsnow = max(clip(fsnow / (1 - O_ice), 0, 1), ice_fraction)
```

For terrain-aware canopy correction, finite sensor azimuth values are
normalized modulo 360 before use. This accepts both the signed `[-180, 180]`
convention used by VIIRS/MODIS source products and an unsigned `[0, 360)`
representation.

Results may use `(y, x)` or `(time, y, x)` dimensions in the lower-level API.
Static ancillary layers broadcast over time. Inputs must already be float32,
aligned, and in their documented physical ranges. NaNs and Dask-backed scene
arrays are preserved.

## Snow radiative products

The independent lower-level functions are:

```python
from spires_postprocess import (
    compute_delta_vis,
    compute_radiative_forcing,
    compute_snow_albedo,
    load_albedo_lookup,
    load_forcing_lookup,
)

albedo_lookup = load_albedo_lookup("albedo.nc")
forcing_lookup = load_forcing_lookup("darkening_and_forcing.nc")

with_albedo = compute_snow_albedo(
    inversion_results,
    cosine_solar_zenith=data.scene["cosine_solar_zenith"],
    cosine_illumination=data.scene["cosine_illumination"],
    lookup=albedo_lookup,
    skyview=data.ancillary.get("skyview"),
    altitude=altitude_in_km,
)
with_delta_vis = compute_delta_vis(
    with_albedo,
    cosine_solar_zenith=data.scene["cosine_solar_zenith"],
    lookup=forcing_lookup,
)
complete = compute_radiative_forcing(
    with_delta_vis,
    cosine_solar_zenith=data.scene["cosine_solar_zenith"],
    lookup=forcing_lookup,
)
```

The albedo operation adds:

- `albedo_clean_flat`
- `albedo_dirty_flat`
- `albedo_clean_terrain_corrected`
- `albedo_dirty_terrain_corrected`

Clean products evaluate the LUT at zero LAP concentration. Dirty products use
the canonical `lap_concentration` result, which must have units `ppm` and
`lap_type="dust"`. Delta-VIS is dimensionless; radiative forcing has units
`W m-2`.

## Lookup contracts

The canonical albedo dataset uses primary variable `albedo`, float32 values,
and exact ordered dimensions:

```text
solar_zenith, illumination_angle, lap_concentration, sqrt_grain_radius
```

Optional `skyview` and `altitude` axes are appended in that order. Coordinate
units are `degrees`, `degrees`, `ppm`, `um^0.5`, `1`, and `km`, respectively.
The LAP coordinate must declare `lap_type="dust"`.

The development delta-VIS/RF dataset uses `delta_vis` and
`radiative_forcing` on the four required canonical dimensions. The same
coordinate and LAP metadata rules apply.

Only NetCDF paths or already-open xarray datasets are accepted. Runtime MATLAB
LUT loading is intentionally unsupported. The current development DV/RF
NetCDF is an offline, zero-soot conversion of the legacy MATLAB values; it is
temporary until scientifically regenerated production tables are available.

Interpolation is linear with no extrapolation. Boundary coordinates are valid;
missing or out-of-domain inputs produce NaN only at affected pixels. All
derived scientific outputs are float32, and same-named derived products are
deterministically replaced without changing base inversion results.
