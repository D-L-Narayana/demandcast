# Methodology

Everything below is implemented from the equations in plain NumPy (`demandcast/models.py`,
`promo.py`, `backtest.py`, `replenish.py`, `evaluate.py`); nothing is delegated to a
forecasting library, so each formula can be traced to a few lines of code and a unit test.

Notation: $y_t$ daily units of one store x SKU series ($t = 1..T$, oldest first), $m = 7$ the
weekly season, $h$ the forecast step ($1..H$), $\hat y_{T+h}$ the point forecast.

## 1. Forecasting models

All models share one interface: `fit(y, promo_flags=None)` then `predict(h, future_flags=None)`
returns $H$ non-negative values (`predict` before `fit` raises `RuntimeError`). Base models
ignore the flag arguments; only the promo wrapper uses them.

### Seasonal naive (`seasonal_naive`)

Repeat the last observed week:

$$\hat y_{T+h} = y_{T+h-7k}, \qquad k = \lceil h/7 \rceil .$$

For series shorter than one week the mean of the available points is repeated.

### Weekday-profiled moving average (`moving_average`)

Level from the trailing window ($w = 28$ days), rescaled by a multiplicative weekday profile:

$$\ell = \frac{1}{w}\sum_{t=T-w+1}^{T} y_t, \qquad
p_d = \frac{\bar y_{(d)}}{\frac{1}{7}\sum_{d'} \bar y_{(d')}}, \qquad
\hat y_{T+h} = \ell \cdot p_{(T+h) \bmod 7},$$

where $\bar y_{(d)}$ is the mean of the observations falling on weekday $d$ (the last eight
weeks when at least eight exist, 0.4.0). The profile is flat when the series is shorter than two
weeks or its mean is zero.

### Holt-Winters, additive, damped trend (`holt_winters`)

Smoothing recursions with level $\ell$, trend $b$, seasonal $s$ and damping $\phi$:

$$\begin{aligned}
\ell_t &= \alpha\,(y_t - s_{t-m}) + (1-\alpha)(\ell_{t-1} + \phi b_{t-1}) \\
b_t    &= \beta\,(\ell_t - \ell_{t-1}) + (1-\beta)\,\phi\, b_{t-1} \\
s_t    &= \gamma\,(y_t - \ell_t) + (1-\gamma)\, s_{t-m} \\
\hat y_{T+h} &= \ell_T + \Big(\sum_{i=1}^{h}\phi^i\Big) b_T + s_{T+h-m(\lfloor (h-1)/m \rfloor + 1)} .
\end{aligned}$$

Initial states come from the first two seasons (level = first-week mean, trend = slope between
the week means, seasonal = first week minus level). Parameters are chosen by grid search over
$\alpha \in \{0.15, 0.3, 0.5\}$, $\beta \in \{0.02, 0.1\}$, $\gamma \in \{0.1, 0.3\}$, $\phi = 0.95$
minimising the in-sample one-step SSE $\sum_t (y_t - \ell_{t-1} - \phi b_{t-1} - s_{t-m})^2$. Since
0.4.0 all twelve candidates are evaluated in one vectorised time loop (`_run_batch`), with results
identical to the per-candidate recursion; `benchmarks/bench.py` keeps the scalar version as the
speed reference.

### Croston with SBA correction (`croston_sba`)

For intermittent demand (at least 50 % zero days, `is_intermittent`), demand sizes $z$ and
inter-demand intervals $q$ are smoothed separately with $\alpha = 0.1$; on each non-zero day
with observed interval $\tau$ since the previous one:

$$z \leftarrow \alpha y_t + (1-\alpha) z, \qquad q \leftarrow \alpha \tau + (1-\alpha) q, \qquad
\hat y = \Big(1 - \frac{\alpha}{2}\Big)\frac{z}{q} .$$

$z$ and $q$ are initialised from the first non-zero observation and the number of days before
it. 0.4.0 fixes the first interval, which 0.3.0 over-counted by one day (D3).

### Theta (`theta`, 0.4.0)

Classical Theta with $\theta = 2$ and weekly multiplicative deseasonalisation:

1. Seasonal indices $I_d = \bar y_{(d)} / \bar y$, clipped to $[0.1, 10]$ (skipped when
   $T < 2m$ or $\bar y = 0$, in which case an all-zero series forecasts zeros);
   $y'_t = y_t / I_{t \bmod 7}$.
2. Theta line 0 = least-squares trend $a + bt$ of $y'$; theta line 2 = $2y' - (a + bt)$, forecast
   by simple exponential smoothing with $\alpha \in \{0.1, 0.2, 0.3, 0.5\}$ (lowest in-sample SSE).
3. Combine $\hat y'_{T+h} = \tfrac12 (a + b(T+h)) + \tfrac12\,\text{SES}_{T+h}$, reseasonalise
   $\hat y_{T+h} = \hat y'_{T+h} \cdot I_{(T+h) \bmod 7}$ and clip at zero.

### Promotion-adjusted wrapper (`promo_<base>`, 0.4.0)

`PromoAdjusted(base)` competes as `promo_moving_average`, `promo_holt_winters` and
`promo_theta`, and only when the series has a promo signal (at least 7 promo days and 28
non-promo days). With flags $f_t \in \{0, 1\}$:

$$\text{lift}_{raw} = \frac{\overline{y}_{f=1}}{\overline{y}_{f=0}}, \qquad
\text{lift} = \operatorname{clip}\Big(1 + (\text{lift}_{raw} - 1)\,\frac{n_p}{n_p + 7},\; 1,\; 5\Big),$$

shrinking towards 1 for few promo days $n_p$ (a zero non-promo mean gives lift 1). The base
model is fitted on the deflated history $y_t / \text{lift}^{f_t}$ and its forecast is multiplied
by $\text{lift}^{f_{T+h}}$ on the future days that carry a scheduled promotion. Without flags the
wrapper equals its base model.

## 2. Backtesting and model selection

**Rolling origin.** For each series and candidate, `n_folds` origins $o_k$ are placed
`horizon` days apart ending at $T - H$ (only origins with at least 56 training days are kept).
The model is fitted on $y_{1..o_k}$ and scored on $y_{o_k+1..o_k+H}$; residuals
$e = y - \hat y$ are pooled over folds. Promo flags are sliced the same way, so the future is
never seen.

**Metrics** (pooled over folds):

$$\text{MAE} = \frac{1}{n}\sum |e|, \qquad
\text{WAPE} = \frac{\sum |e|}{\sum |y|}\ (\text{undefined when } \sum|y| = 0), \qquad
\text{bias} = \frac{1}{n}\sum (\hat y - y),$$

$$\text{MASE} = \frac{\text{MAE}}{\frac{1}{n_{train}-7}\sum_{t=8}^{n_{train}} |y_t - y_{t-7}|}
\quad (\text{scale from the first fold's training window; undefined when the scale is } 0).$$

$\sigma_{res}$ = sample standard deviation of the pooled residuals (feeds safety stock).

**Selection.** Candidates are the registry order (`seasonal_naive`, `moving_average`,
`holt_winters`, `theta`, `croston_sba` only if intermittent) plus the `promo_*` wrappers when the
series has promo signal, optionally restricted with `--models`. The winner minimises the
criterion (`mae` by default; `wape` or `mase` since 0.4.0); undefined metrics sort last; ties
are broken by $|\text{bias}|$ and then by name, so selection is deterministic.

## 3. Prediction intervals

0.3.0 used a constant symmetric band $\hat y \pm 1.2816\,\sigma_{res}$ (nominal 80 %) that was
never verified. Since 0.4.0 the interval comes from the empirical residual quantiles of the
winning model (`--interval LEVEL`, default 0.8, `--interval-method empirical`):

$$q_{lo} = \min\big(0, Q_{(1-L)/2}(e)\big), \qquad q_{hi} = \max\big(0, Q_{1-(1-L)/2}(e)\big),$$

$$\text{lower}_h = \max\big(0,\ \hat y_h + q_{lo}\,(1 + g\,(h-1))\big), \qquad
\text{upper}_h = \hat y_h + q_{hi}\,(1 + g\,(h-1)),$$

with a horizon growth factor $g = \operatorname{clip}\big((r_2 / r_1 - 1) / (H/2),\ 0,\ 0.03\big)$,
where $r_1$ and $r_2$ are the mean absolute residuals over the first and second half of the
horizon steps ($g = 0$ with fewer than 20 residuals, $r_1 = 0$ or $H < 4$). The clamps guarantee
$0 \le \text{lower} \le \hat y \le \text{upper}$. `--interval-method normal` restores the symmetric
$\pm z_{L}\,\sigma_{res}$ band. Realised coverage is measured by `evaluate` (section 6).

## 4. Replenishment: periodic review $(R, s, S)$

With supplier lead time $L$ days, review period $R$ days (`--review-period`), cycle service
level $p$ (`--service-level`, or per ABC class with `--service-level-by-class`), $z_p = \Phi^{-1}(p)$
and inventory position $IP = \text{on\_hand} + \text{on\_order}$:

$$\begin{aligned}
\text{LTD} &= \sum_{h=1}^{L} \hat y_h & \text{(lead-time demand)} \\
\text{SS}  &= z_p\,\sigma_{res}\,\sqrt{L + R} & \text{(safety stock)} \\
s &= \text{LTD} + \text{SS} & \text{(reorder point)} \\
S &= \sum_{h=1}^{L+R} \hat y_h + \text{SS} & \text{(order-up-to level)} \\
q &= \begin{cases} \lceil (S - IP)/c \rceil \cdot c & IP < s \\ 0 & IP \ge s \end{cases} & \text{(case pack } c\text{)}
\end{aligned}$$

Perishables are capped so that no more is ordered than can sell within the shelf life
($\lfloor \sum_{h \le \text{shelf life}} \hat y_h \rfloor - IP$, rounded up to the pack). Since
0.4.0 a minimum order quantity rounds up and a maximum rounds *down* to the pack after the
case-pack rounding; the shelf-life cap always wins. The model horizon is extended to
$\max(H, \max_p L_p + R)$ internally so that $S$ is always covered.

## 5. Stock-out risk, expected shortfall, priority and budget (0.4.0)

Demand over the protection interval is treated as normal with
$\mu_{LR} = \sum_{h=1}^{L+R} \hat y_h$ and $\sigma_{LR} = \sigma_{res}\sqrt{L+R}$;
with $k = (IP - \mu_{LR}) / \sigma_{LR}$:

$$P(\text{stock-out}) = 1 - \Phi(k), \qquad
\mathbb{E}[\text{shortfall}] = \sigma_{LR}\,\big[\varphi(k) - k\,(1 - \Phi(k))\big]
\quad\text{(the normal loss function)},$$

$$\text{priority} = \mathbb{E}[\text{shortfall}] \times \text{unit cost}, \qquad
\text{days of cover} = \frac{IP}{\mu_{LR} / (L+R)} .$$

When $\sigma_{LR} = 0$ the risk is 1 if $IP < \mu_{LR}$ else 0 and the shortfall is
$\max(0, \mu_{LR} - IP)$. Sanity checks: at $IP = \mu_{LR}$ the risk is 0.5 and the shortfall is
$0.3989\,\sigma_{LR}$; both decrease monotonically in $IP$.

**Order budget** (`--order-budget B`): orders are sorted by priority (descending, ties by
store/product id) and approved whole while $\text{qty} \times \text{unit cost}$ fits into the
remaining budget (first fit); the rest are deferred with `order_qty = 0`, `requested_qty` kept and
the reason annotated. $\sum \text{approved cost} \le B$ always holds.

**ABC classes** use cumulative revenue share over the trailing 90 days before the cutoff:
A up to 70 %, B up to 90 %, C beyond (computed as of the cutoff, so backdated runs do not leak).

## 6. Realised-accuracy evaluation (0.4.0)

For a run with cutoff $c$ and horizon $H$, `evaluate` joins `forecasts` with `sales_daily` for
$c < \text{day} \le \min(c + H, \text{last sales day})$ and computes per series: realised MAE,
WAPE, bias (as in section 2) and interval coverage

$$\text{coverage} = \frac{1}{n}\sum_{h} \mathbf 1\big[\text{lower}_h \le y_{c+h} \le \text{upper}_h\big].$$

Aggregates: MAE and bias are averaged over series, WAPE is $\sum|e| / \sum|y|$ over all
series-days, coverage is weighted by the number of evaluated days. A nominal 80 % interval should
show coverage near 0.8; stock-out days are counted separately because censored sales bias the
actuals downwards. Results are persisted per run in `forecast_evaluations` and exposed by the
`evaluation_summary`, `evaluation_history` and `forecast_vs_actual` queries.

## 7. Synthetic data generator

Demand is multiplicative, $\lambda_t = \text{base}_{s,p}\cdot w_{dow}\cdot y(t)\cdot \text{trend}(t)
\cdot \text{promo}(t)\cdot \text{holiday}(t)$ with Poisson noise, and sales are *censored* by an
$(s, S)$ inventory simulation so stock-outs appear as in real POS data (`stockout_flag = 1`,
`units_sold < demand`). The generator is seeded; the default dataset is byte-identical across
releases (guarded by a hash test), new features (`--snapshot-every`, `--future-promo-days`) draw
their random numbers after the sales loop.
