# Data

## First party data

Directly from the S3 bucket.
<!-- NOTE: `flightlist_train.parquet` renamed to `flight_list_train.parquet` for consistency -->
### 1. Fuel Data (`fuel_*.parquet`)

133,984 rows (train)

24,972 rows (rank)

| Column      | Units | Type                    | Description                                                    |
| ----------- | ----- | ----------------------- | -------------------------------------------------------------- |
| `idx`       | -     | INTEGER                 | Unique row identifier                                          |
| `flight_id` | -     | VARCHAR                 | Links to the flight list and trajectory                        |
| `start`     | UTC   | TIMESTAMP WITH TIMEZONE | The start timestamp of the interval                            |
| `end`       | UTC   | TIMESTAMP WITH TIMEZONE | The end timestamp of the interval                              |
| `fuel_kg`   | kg    | DOUBLE                  | The target variable. In submission files, this is set to `0.0` |

!!! note
    The `start` and `end` often correspond to the timestamps of ACARS reports.

!!! warning
    `fuel_kg` target variable has quantisation artifacts: data is not a simple continuous distribution but a composite from at least two distinct sources: imperial (pounds) and metric (kilograms) units with a 2sf rounding step.

### 2. Flight List (`flight_list_*.parquet`)

124,094,050 total rows (train, 11088 files, 3.2G)

24,499,924 total rows (rank, 1929 files, 616M)

| Column             | Units | Type      | Description                           |
| ------------------ | ----- | --------- | ------------------------------------- |
| `flight_id`        | -     | VARCHAR   | A unique identifier for the flight    |
| `flight_date`      | -     | DATE      | The date of the flight                |
| `takeoff`          | UTC   | TIMESTAMP | The timestamp of takeoff              |
| `landed`           | UTC   | TIMESTAMP | The timestamp of landing              |
| `origin_icao`      | -     | VARCHAR   | ICAO code for the departure airport   |
| `origin_name`      | -     | VARCHAR   | Name of the departure airport         |
| `destination_icao` | -     | VARCHAR   | ICAO code for the destination airport |
| `destination_name` | -     | VARCHAR   | Name of the destination airport       |
| `aircraft_type`    | -     | VARCHAR   | ICAO code for the aircraft model      |

### 3. Trajectories (`flights_*/<flight_id>.parquet`)

Time-series state vectors for each flight: trajectories may be incomplete and contain anomalies.

| Column          | Units   | Type         | Description                                      |
| --------------- | ------- | ------------ | ------------------------------------------------ |
| `timestamp`     | UTC     | TIMESTAMP_NS | Timestamp of the position report                 |
| `flight_id`     | -       | VARCHAR      | Links to the flight list and fuel data           |
| `typecode`      | -       | VARCHAR      | Aircraft type code                               |
| `latitude`      | degrees | DOUBLE       | Position latitude in decimal degrees             |
| `longitude`     | degrees | DOUBLE       | Position longitude in decimal degrees            |
| `altitude`      | ft      | DOUBLE       | Altitude                                         |
| `groundspeed`   | knots   | DOUBLE       | Ground speed                                     |
| `track`         | degrees | DOUBLE       | Track angle                                      |
| `vertical_rate` | ft/min  | DOUBLE       | Rate of climb/descent                            |
| `mach`          | -       | DOUBLE       | Mach number (may be NULL)                        |
| `TAS`           | knots   | DOUBLE       | True airspeed (may be NULL)                      |
| `CAS`           | knots   | DOUBLE       | Calibrated airspeed (may be NULL)                |
| `source`        | -       | VARCHAR      | The origin of the data, either `adsb` or `acars` |

!!! tip "A Note on `source`"
    Data from `adsb` and `acars` have different characteristics. `acars` data, for instance, may include `mach`, `TAS`, and `CAS`, which are not present in standard ADS-B reports.

### 4. Airports (`apt.parquet`)

8787 rows

| Column      | Units   | Type    | Description                            |
| ----------- | ------- | ------- | -------------------------------------- |
| `icao`      | -       | VARCHAR | ICAO code of the airport (e.g. `VHHH`) |
| `latitude`  | degrees | DOUBLE  | Airport latitude coordinate            |
| `longitude` | degrees | DOUBLE  | Airport longitude coordinate           |
| `elevation` | ft      | DOUBLE  | Airport elevation (may be NULL)        |

## Submission Format

Final predictions must be submitted as a single Parquet file.

### File Format

The submission file must contain the exact same entries as the provided `fuel_rank_submission.parquet` or `fuel_final_submission.parquet` file, with the `fuel_kg` column populated with the predictions.

The file must contain two columns:

- `idx`: The row identifier, matching the submission template.
- `fuel_kg`: Predicted fuel consumption (kilograms) for the given interval.

example:

```text
idx   flight_id     start                end                  fuel_kg
---   -----------   -----                ---                  -------
0     prc770822360  2025-04-13 04:31:04  2025-04-13 05:01:04  250.3
1     prc770822360  2025-04-13 05:01:04  2025-04-13 05:16:04  120.8
2     prc770822360  2025-04-13 05:16:04  2025-04-13 05:46:04  2500.0
...   ...           ...                   ...                 ...
```

!!! important "Missing Rows"
    Only the `idx` and `fuel_kg` columns are used for scoring. If the submission is missing rows that are present in the template, they will be assigned a value of `0.0`, which may negatively impact your RMSE score.
