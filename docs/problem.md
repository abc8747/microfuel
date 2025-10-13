# Problem

The problem is to predict the integrated fuel burn, $\int_{t_{\text{start}}}^{t_{\text{end}}} \dot{m_f}(t) \, dt$, over a specified time interval. The challenge is to learn a model of the aircraft's continuous-time dynamics from a sparse, irregularly-sampled observation history, $X = \{(x_i, t_i)\}_{i=1}^N$. The prediction target is defined over a sub-interval $[t_{\text{start}}, t_{\text{end}}]$ which only covers a fraction of the total observation window.

A successful model must address four key challenges:

1. Learn a continuous latent state trajectory $z(t)$ that accurately represents the aircraft's physical state from the discrete observation set $X$.
2. Robustly handle non-uniform time gaps $\Delta t_i$ between observations.
3. Capture long-range, non-linear dependencies inherent in flight data.
4. Directly and efficiently compute the target integral, where the instantaneous fuel burn rate is a function of the latent state, $\dot{m_f}(t) = g(z(t))$.

This problem structure allows for two inference paradigms. Given all observations $X_{1:N}$, estimating the state $z(t)$ for $t \in [t_{\text{start}}, t_{\text{end}}]$ can be framed as:

- **Filtering:** Using only past and current data, $p(z(t) | X_{t_i \le t})$. This is a causal approach.
- **Smoothing:** Using the entire observation history, $p(z(t) | X_{1:N})$. This is more powerful for offline analysis as it uses "future" data (relative to the segment) to correct state estimates, likely yielding a more accurate representation.

## TODO

- [ ] tackle anomalous data points (sudden jumps in altitude or speed)

third party sources of information

- [ ] google-arco era5 for isa deviation, GS -> TAS conversion, temperature, pressure
- [ ] kinematic features ($p, q, r$, $\dot{x}, \dot{y}, \dot{z}$)
- [ ] specific energy (potential + kinetic)
- [ ] <https://github.com/DGAC/Acropole>
