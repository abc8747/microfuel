# microfuel

[Documentation](https://abc8747.github.io/microfuel)

`microfuel` is a lightweight, end-to-end differentiable surrogate model for aircraft fuel consumption.

While the prevalent approach involves using Gradient Boosted Decision Trees (LightGBM/XGBoost) fed by extensive manual feature engineering and external data, they are non-differentiable.

We ask the question: How accurate can we be at fuel flow prediction using raw kinematics alone, while maintaining the smoothness required for gradient-based trajectory optimisation?

## Performance

We strongly prioritise simplicity over marginal gains in predictive accuracy derived from exogenous data dependencies.

- Inputs: `aircraft_type`, `flight_duration` and smoothed time series of {`altitude`, `groundspeed`, `vertical_rate`}.
- Output: Average fuel burn rate $\dot{m}_f$ (kg/s) in a specified segment.

| Model                     | Parameters | Test RMSE   | Capability                                                                            |
| :------------------------ | :--------- | :---------- | :------------------------------------------------------------------------------------ |
| `microfuel-v1.0-realtime` | 20,021     | 238.92 kg   | Only relies on information within the segment. Suitable for real-time inference.      |
| `microfuel-v1.0-offline`  | 66,689     | 222.54 kg ¹ | Processes the entire flight history. Suitable for high-fidelity post-flight analysis. |

¹ As of 2025-11-10, this model is ranked #4 out of 164 teams in [Phase 1 of the PRC Data Challenge 2025](https://ansperformance.eu/study/data-challenge/dc2025/ranking.html). No changes to the model were made in Phase 2.

For a detailed comparison of the architecture and methodology against SOTA models, see the [comparison section of our documentation](https://abc8747.github.io/microfuel/comparison).

## Installation

The repository is currently in a **pre-alpha state and not ready for production use**. While the ultimate goal is a NumPy-only inference engine for maximum portability, the current training code relies on `torch` and `triton`.

By the end of December 2025, the code is expected to be stabilised, with model weights released and a PyPI package published.

For a step-by-step reproduction guide, refer to the [Quickstart section](https://abc8747.github.io/microfuel/quickstart) of our documentation.

A [`tangram` plugin](https://github.com/open-aviation/tangram) will also be developed to demonstrate real-time fuel inference from live [`jet1090` data](https://github.com/xoolive/rs1090).
