# microfuel

`microfuel` is a high-performance, lightweight surrogate model for aircraft fuel consumption, trained using open ADS-B and ACARS data.

It does not try to be the SOTA in prediction accuracy, but is developed with simplicity in mind:

- no costly 4D weather grids evaluations required
- no aircraft mass required
- inputs: `aircraft_type`, smoothed time series of `altitude`, `groundspeed`, `vertical_rate` only.
- output: cumulative fuel burn (kg) within the specified segment.

Available model flavours:

| Model                     | Number of Params | Test RMSE   | Description                                                                            |
| :------------------------ | :--------------- | :---------- | :------------------------------------------------------------------------------------- |
| `microfuel-v1.0-realtime` | 20,021           | 238.92 kg   | Sees only the specific flight segment being queried. Suitable for real-time inference. |
| `microfuel-v1.0-offline`  | 66,689           | 222.54 kg ¹ | Processes the entire flight history. Suitable for high-fidelity post-flight analysis.  |

¹ As of 2025-11-10 (end of phase 1), this model is ranked #4 out of 164 teams in the [PRC Data Challenge 2025](https://ansperformance.eu/study/data-challenge/dc2025/ranking.html).

## Installation

The ultimate goal of this repository will be implement a numpy-only inference engine for maximum portability and minimal depedencies.

However, the current repository is in a **pre-alpha state and not yet ready for production use**. The inference engine currently depends on heavy dependencies like `torch` and `triton`. Significant maintenance and breaking changes of this repo is planned. Model weights will be released once things have stabilised (hopefully by mid December 2025).

For an step-by-step explanation to reproduce the model, refer to the [quickstart in the documentation](https://abc8747.github.io/microfuel).

A [`tangram` plugin](https://github.com/open-aviation/tangram) will also be created to demonstrate real-time fuel inference from [`jet1090` data](https://github.com/xoolive/rs1090).
