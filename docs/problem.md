# Problem Formulation

The objective is to estimate the total [mass][isqx.MASS] of fuel consumed, $m_f$, over a specific flight segment defined by the time interval $[t_{\mathrm{start}}, t_{\mathrm{end}}]$.

The fuel mass flow rate $\dot{m}_f$ is a function of the [thrust][isqx.THRUST] required to maintain the aircraft's kinematic state. The simplified governing equation for longitudinal motion of a point-mass aircraft is:
$$
m \dot{V} = T - D - mg \sin \gamma,
$$
where:

- $m$: Aircraft [mass][isqx.MASS] (time-varying: $m(t) = m_0 - \int_0^t \dot{m}_f(\tau) d\tau$). Note that the [initial takeoff mass][isqx.aerospace.TAKEOFF_MASS] is unknown.
- $\dot{V}$: Rate of change of the [true airspeed][isqx.aerospace.TAS] along the flight path.
- $T$: [Thrust][isqx.THRUST].
- $D$: [Drag][isqx.aerospace.DRAG] ($D = \frac{1}{2}\rho V^2 S C_D$).
- $g$: [Gravitational acceleration][isqx.ACCELERATION_OF_FREE_FALL].
- $\gamma$: Flight path angle.

The instantaneous fuel flow is related to thrust via the [thrust specific fuel consumption][isqx.aerospace.TSFC] $c_T$: $\dot{m}_f = c_T T$, which varies with the aircraft's altitude and mach number ([Bartel and Young, 2008](https://arc.aiaa.org/doi/10.2514/1.35589))

## Identifiability

A challenge in using public surveillance data (ADS-B) is the non-identifiability of the kinetic state.

$$
T - D = m(\dot{V} + g \sin \gamma)
$$

The RHS contains kinematic terms ($\dot{V}, \gamma$) which are observable from surveillance data (see [`microfuel.datasets.raw.TrajectoryRecord`][]), and mass $m$, which is unknown but bounded. However, the LHS represents the Excess Thrust. Without an aerodynamic model (e.g. [`openap`](https://github.com/junzis/openap) or [BADA](https://www.eurocontrol.int/model/bada)) to determine $D$, it is not possible to isolate $T$ from $T - D$ given only kinematic observations.

Since we cannot explicitly solve for $T$ to compute $\dot{m}_f$, we learn a mapping function $\mathcal{F}_\theta$ parameterised by a neural network that maps a sequence of observable kinematic states to the cumulative fuel burn:

$$
\int_{t_\mathrm{start}}^{t_\mathrm{end}} \dot{m}_f dt \approx \mathcal{F}_\theta(X_{t_\mathrm{start}:t_\mathrm{end}} | \text{type}_\text{ac})
$$

## Model Architecture

We utilise a Gated Delta Network (GDN) ([Yang et al., 2024](https://arxiv.org/abs/2412.06464)), a simplified state-space model (SSM) that offers the inference speed of RNNs with the expressive capacity of Transformers.

GDN introduces an input-dependent state transition while maintaining $O(L)$ linear complexity. The update rule for the memory state $S_t \in \mathbb{R}^{d_k \times d_v}$ is:

$$
S_t = S_{t-1} \underbrace{(\alpha_t I - \beta_t k_t k_t^\top)}_{\text{Decay \& Erase}} + \underbrace{\beta_t v_t k_t^\top}_{\text{Write}}
$$

### Control-theoretic interpretation

The GDN update rule can be rewritten to reveal its operation as an online gradient descent optimiser solving a least squares problem:
$$
S_t = \alpha_t S_{t-1} + \beta_t \underbrace{(v_t - S_{t-1} k_t)}_{\text{Innovation / Error}} k_t^\top
$$

1. The model queries the current memory state $S_{t-1}$ with key $k_t$ to predict a value $\hat{v}_t = S_{t-1}k_t$.
2. It calculates the innovation, or the difference between the actual information $v_t$ and the retrieved information.
3. It updates the memory matrix $S$ by moving it in the direction of the error, weighted by the step size $\beta_t$.

This effectively makes the GDN a "Fast Weight Programmer". The memory $S_t$ is a dynamic linear model that is continuously re-optimised (trained) at every time step to minimise the reconstruction error of the incoming flight data stream.

### Inference Modes: Realtime vs. Offline

We address the unknown mass $m(t)$ by leveraging different temporal contexts depending on the inference scenario.

#### Realtime

In the realtime model, prediction depends strictly on causal information $P(y_t | x_{0:t})$. The model must infer the current fuel burn rate based solely on the kinematic history up to the current moment.

#### Offline

For post-flight analysis, we employ an offline model that utilizes the entire flight trajectory $x_{0:T}$ to predict the fuel burn of a specific segment $y_{t:t+\Delta}$.

The model processes two streams via [`microfuel.model.FuelBurnPredictor`][]:

1. Segment Stream: High-resolution kinematics of the query interval.
2. Flight Context Stream: The full trajectory from takeoff to landing.

We hypothesise that the flight context stream allows the model to implicitly learn a latent representation of the aircraft's mass and drag constraints. For example, the rate of climb at takeoff reveals the initial weight, while the total energy dissipation during descent constrains the drag polar. This global context corrects the local fuel burn estimates.

### Hypernetworks for Physics Adaptation

Aircraft performance characteristics vary drastically across different aircraft types (e.g. A380 vs A320). A standard embedding approach is insufficient because it only learns to shift the *bias* of the features.

We employ a Static Hypernetwork ([`microfuel.model.StaticHyperNet`][], [Ha et al., 2016](https://arxiv.org/abs/1609.09106)) to address this. Instead of sharing weights across all aircraft, the model learns a manifold of parameters. Let $e_{ac} \in \mathbb{R}^{d_{emb}}$ be the embedding for a specific aircraft type. A small MLP $\mathcal{H}$ generates the weights for the input projection layers of the main network:

$$
\begin{align*}
\theta_\text{proj}^{\text(ac)} &= \mathcal{H}(e_\text{ac}) \\
h_t &= W_\text{proj}^{\text(ac)} x_t + b_\text{proj}^{\text{(ac)}}
\end{align*}
$$

### Class Imbalance

The aircraft type distribution follows a power law, causing the model to be biased towards common aircraft types (e.g., A320, A20N) while performing poorly on rare types (e.g., MD11, A318).

We introduce an optional Class-Balanced (CB) loss ([Cui et al., 2019](https://arxiv.org/abs/1901.05555)), weighing the loss based on the "effective number of samples", which posits that the marginal benefit of new data diminishes as sample size increases due to information overlap.

The effective number of samples $E_n$ for a class with $n$ samples is defined as $E_n = (1 - \beta^n) / (1 - \beta)$, where $\beta \in [0, 1)$ is a hyperparameter. The CB loss weights the loss function $\mathcal{L}$ for a sample of class $y$ by the inverse of its effective frequency:

$$
\mathcal{L}_{\text{CB}}(\mathbf{p}, y) = \frac{1 - \beta}{1 - \beta^{n_y}} \mathcal{L}(\mathbf{p}, y)
$$

This up-weights the loss for rare classes, encouraging the model to learn their physics effectively despite the data sparsity.
