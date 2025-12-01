# Comparison

This section compares the architecture and feature engineering techniques against several SOTA models in the [PRC Data Challenge 2025](https://ansperformance.eu/study/data-challenge/dc2025/ranking.html).

## unique-guitar (this repository)

Phase 2 Test RMSE: 222.54 kg (rank: 7)

Source Code: <https://github.com/abc8747/microfuel>

Architecture: gated delta network

Extra Data/Libraries: --

Inputs:

- Kinematic sequence: `altitude`, `groundspeed`, `vertical_rate` (smoothed via Kalman filter).
- Context: `flight_progress` (normalized time), `flight_duration`, `aircraft_type` (embedding index).

Methodology:

- Signal processing: Kalman filter + RTS smoother for trajectory denoising (constant velocity assumption).
- Architecture: GDN (linear attention variant) with dual-stream processing:
    1. Segment-level: Local kinematics.
    2. Flight-level: Full trajectory context (implicit mass identifiability).
- Adaptation: Static Hypernetwork generates weights based on `aircraft_type` embedding.
- Loss function: Class-Balanced (CB) loss to upweight rare aircraft types; optimisation on RMSE of fuel *rate*, converted to total kg.

## resourceful-quiver

Phase 2 Test RMSE: 199.91 kg (rank: 1)

Source Code: <https://github.com/meldorr/PRC-2025>

Architecture: LightGBM Regressor (two-stage: mass estimation -> fuel prediction)

Extra Data/Libraries: ERA5 weather (u/v wind, temp, humidity), `ac_tows.csv` (aircraft weights), `openap` (library).

Inputs:

- Flight Context: `seg_duration`, `seg_dist`, `flight_duration`, `full_flight_dist`, `aircraft_type`, `phase`, `m_tow`, `oew`.
- Trajectory Aggregates: `mean`/`std` for `groundspeed`, `track`, `vertical_rate`, `mach`, `TAS`, `CAS`; `altitude_mean`, `vertical_rate_min`, `vertical_rate_max`.
- Mass Features: `tow_est_kg` (predicted via separate LGBM using climb profile/vertical acceleration), `mass_est_tf_mean`, `mass_est_tf_std` (time-flown based mass decay).
- Derived Physics: `ff_kgs_est_mass_tf_mean`, `ff_kgs_est_mass_tf_std` (OpenAP fuel flow estimate using decaying mass).
- Binned Statistics: `vertical_rate_mean_{0..9}`, `vertical_rate_std_{0..9}`.

Methodology:

- Preprocessing: LCC projection resampling (1s), Savitzky-Golay filtering, custom phase detection, altitude gap filling.
- Aerodynamics: TAS calculated from Mach/CAS and ERA5 wind components; mass estimated via climb performance.
- Modelling: LightGBM on ~48 features, 5-fold CV, target `ff_kgs` (fuel flow kg/s) converted to total fuel.

## bright-lobster

Phase 2 Test RMSE: 213.24 kg (rank: 2)

Source Code: <https://github.com/eeftychiou/PRCXGBoost>

Architecture: XGBoost Regressor ensemble (top 10 Models)

Extra Data/Libraries: SkyVector (runway heading/length/elevation), IATA market reports (load factors), METAR weather, `openap` (library), `pygeomag` (library).

Inputs:

- Airport/runway: `origin_` / `destination_` `longitude`, `latitude`, `elevation`, `RWY_{1..8}_{HEADING/LENGTH/ELEVATION}`.
- Aircraft Meta: `mfc`, `pax_high`, `fuselage_height`, `wing_mac`, `wing_t/c`, `flaps_{type/area/bf_b/Sf_S}`, `cruise_mach`, `engine_default`, `drag_{cd0/e/gears}`, `fuel_fuel_coef`, `limits_OEW`.
- Weather (METAR): `dep_` / `arr_` `tmpf`, `sknt`, `vsby`, `wx_intensity`; boolean flags for `thunderstorm`, `freezing`, `shower`, `rain`, `snow`, `fog_mist`, `haze_smoke`.
- Flight Context: `great_circle_distance_km`, `flight_duration_seconds`, `average_load_factor` (IATA), `estimated_payload_kg`, `estimated_takeoff_mass`, `estimated_total_fuel_kg`, `trip_fuel_kg`, `contingency_fuel_kg`, `final_reserve_fuel_kg`.
- Time: `seg_start_day_of_week`, `seg_{start/end}_time_decimal`, `flight_{start/end}_day_of_week`, `flight_{start/end}_time_decimal`, `seg_{end/start}_to_{landing/takeoff}`.
- Trajectory Aggregates: `min`/`max`/`mean`/`std`/`delta` for `latitude`, `longitude`, `altitude`, `groundspeed`, `track`, `vertical_rate`, `mach`, `TAS`, `CAS`, `calculated_speed`, `vertical_rate_change`, `dist_to_origin_km`, `dist_to_dest_km`; `start_alt_rev`, `end_alt_rev`, `alt_diff_rev`, `alt_diff_rev_std`, `mean_time_in_air`.
- Phases: `phase_fraction_{climb/cruise/descent/approach/gnd/level/na}`, `ee_phase_duration_{parked/taxi_out/takeoff/climb/cruise/descent/approach/landing/taxi_in}`.
- Derived Physics: `fuel_consumption_{gnd/cl/de/lvl/cr/na}`, `fuel_consumption` (sum), `seg_avg_burn_rate`.
- Interactions: `duration_x_{mass/altitude}`, `distance_x_mass`, `alt_x_mass`, `speed_x_mass`; polynomials (`segment_duration_{sq/cub}`, `phase_duration_cl_{sq/cub}`, `alt_diff_rev_sq`).

Methodology:

- Data Enrichment: Web scraping SkyVector, mapping IATA load factors to routes.
- Augmentation: Generation of 25k synthetic widebody samples via Gaussian noise injection on long segments.
- Feature Selection: Sequential Feature Selection (SFS) with XGBoost base.
- Training: RandomizedSearchCV, ensemble of top 10 validation Models trained on 100% data (train+synthetic).

## sincere-glacier

Phase 2 Test RMSE: 214.36 kg (rank: 3)

Source Code: <https://github.com/johntad110/sincere-glacier-prc2025>

Architecture: Hybrid Stacking Ensemble (LightGBM + LSTM -> Ridge Regression)

Extra Data/Libraries: --

Inputs:

- GBM features (aggregated): `duration`, `n_points`, `total_dist`, `time_since_takeoff`, `time_to_landing`, `relative_time`, `od_distance`; `origin_` / `dest_` `lat`, `lon`, `elev`; `aircraft_type`.
  - Statistics (`avg`/`std` and `min`/`max` where applicable): `avg_alt`, `avg_speed`, `avg_vertical_rate`, `avg_acc`, `avg_energy_rate`.
  - Aerodynamics: `avg_mach`, `avg_dynamic_pressure`, `avg_air_density`, `avg_parasitic_power`, `avg_induced_power`, `avg_climb_power`.
  - Physics: `mass_proxy` (estimated from climb Newtonian dynamics).
- LSTM features (sequential): `altitude`, `groundspeed`, `vertical_rate`, `sin_track`, `cos_track`, `dist_step`, `acceleration`, `energy_rate` (sequence length 32).
  - Static context: `aircraft_type` (embedding), `duration`, `total_dist`.

Methodology:

- Preprocessing: Z-score normalisation, ISA standard atmosphere Modelling, outlier filtering based on fuel flow limits.
- Mass Estimation: Physics-based proxy derived using thrust-drag equation during climb.
- Stacking: Non-negative Ridge Regression combining Out-of-Fold (OOF) predictions from physics-aware GBM and sequence-aware LSTM.

## wise-watermelon

Phase 2 Test RMSE: 215.68 kg (rank: 4)

Source Code: <https://github.com/isaacOluwafemiOg/prc2025_wisewatermelon>

Architecture: CatBoost Regressor Ensemble

Extra Data/Libraries: FAA aircraft characteristics, World Airports CSV, `openap` (library), `acropole` (library), `traffic` (library).

Inputs:

- Scalars: `aircraft_type`, `engine_Model`, `wake_category`; `flight_fuel` (total estimated), `fl_max_alti`, `real_flight_dur`, `missing_segment`.
- Segment aggregates (`sum`, `mean`, `min`, `max`, `start`, `end`, `nancount`): `altitude`, `groundspeed`, `mach`, `CAS`, `fuel_flow` (OpenAP), `fuel` (OpenAP), `drag` (OpenAP), `thrust` (OpenAP), `cl_fuel`, `enr_fuel`, `dist_from_ades`, `acp_fuel` (Acropole), `acp_fuelflow` (Acropole).
- Derived: `total_climb_height`, `unscaled_approx_seg_fuel`, `resample_quality_score`.
- Phases: `seg_{phase}_dur` (for GND, CL, DE, LVL, CR, NA), `flight_{phase}_ct`, `all_seg_phase_dur`.

Methodology:

- Preprocessing: Trajectory merging (handling duplicates), synthetic row injection for coverage, linear interpolation/resampling to 60s.
- Physics: Vectorised Haversine, ATM phase labelling, OpenAP/Acropole estimation per timestamp for fuel/drag/thrust.
- Validation: Stratified Group K-Fold (k=5) grouped by `flight_id`.
- Modelling: CatBoost with Optuna tuning and physics-constrained clipping (non-negative).
