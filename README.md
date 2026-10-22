# Cross-Sectional Equity Alpha Research

**Can daily price and volume data predict which stocks will outperform their peers tomorrow?**

An end-to-end quantitative research study of **next-day cross-sectional equity returns** using daily open, high, low, close, and volume (OHLCV) data. The project tests two economically motivated signals, **intermediate-term momentum** and **short-horizon reversal**, then asks whether more features or machine learning add *incremental, tradable* value.

The central result is deliberately two-sided: a simple rank-based signal has a small, positive historical information coefficient, and gradient boosting improves rank prediction, **but higher predictive IC does not automatically lead to a better long-short portfolio**. Most predictive information appears in the overnight return, where implementation costs become particularly important.

**Research notebook:** [`Cross-Section-Equity-Research.ipynb`](Cross-Section-Equity-Research.ipynb)  
**Research presentation:** [`Project-Research.pdf`](Project-Research.pdf)

## Key results

| Research question | Finding |
|---|---|
| Does the transparent momentum/reversal signal predict next-day relative returns? | **+0.0131 mean walk-forward Spearman IC**, 95% block-bootstrap CI **[+0.0083, +0.0178]** across **4,943 trading dates**. |
| Do momentum and reversal provide complementary information? | Standalone IC: **+0.0129** momentum, **+0.0071** reversal; signal-rank correlation **−0.017**. Their equal-weight blend has **+0.0132** full-sample IC and **1.20 IC information ratio**. |
| Does ML improve prediction? | On **4,692 matched evaluation dates**, the raw IC increases from **0.0124** (simple blend) to **0.0159** (conditioned Ridge) and **0.0173** (gradient boosting). |
| Does the better ML predictor make more money? | **Not in the tested portfolio.** In the matched ML portfolio comparison, the blend's gross Sharpe is **0.566**, versus **0.141** for the boosted model. |
| When is the signal realized? | The blend's IC is **+0.0344 overnight** (close → next open), versus **−0.0001 intraday** (next open → close). |
| Can it survive transaction costs? | The continuously held baseline has gross Sharpe **0.539**, turnover **0.566/day**, and approximately **4.57 bps per dollar traded** of break-even cost. A daily-flattened overnight book has higher gross Sharpe (**1.543**) but only **1.83 bps** of break-even cost. |

*These are historical research results, not live performance. Full-sample descriptive IC, walk-forward IC, and matched-model portfolio results use different evaluation samples and should not be compared as if they were one backtest.*

## Research motivation

Daily price and volume are widely available, but a statistically significant forecast is not necessarily an economically viable trade. This study follows the chain from **hypothesis → data validity → signal construction → out-of-sample evaluation → risk exposures → portfolio economics**.

Two hypotheses motivate the baseline:

1. **Momentum / gradual repricing:** information may be incorporated into prices over multiple months, causing relative winners to continue outperforming.
2. **Short-horizon reversal / temporary price pressure:** transient liquidity demand may push prices away from short-run equilibrium, with partial reversal afterward.

OHLCV can test the *implications* of these mechanisms. It cannot identify investor motives or prove a causal liquidity-provision premium.

## Dataset and research design

| Dimension | Specification |
|---|---|
| Data | Stooq daily OHLCV, assembled into a prebuilt Parquet panel |
| Raw coverage | **20,609,355 stock-date rows**, **9,530 vendor tickers**, 1962–September 2026 |
| Research sample | 2005–September 18, 2026 |
| Daily universe | Top **1,000 eligible names** by **lagged 60-observation median dollar volume** (at least 40 valid observations) |
| Typical coverage | Approximately **990 names/day**; **5.40 million** in-universe stock-days; **97.7%** baseline signal coverage |
| Prediction target | Next adjacent market session's close-to-close **stock return minus observed cross-sectional mean** |
| Core evaluation | Daily Spearman rank IC; monthly historical-weight walk-forward; date-block bootstrap confidence intervals |
| Portfolio | Daily top/bottom signal deciles, dollar-neutral, inverse-volatility weighted, unit gross exposure |

### Data integrity and leakage controls

- **Preserve the time axis.** Flag suspect price/return observations rather than deleting interior dates and accidentally treating a multi-session gap as a one-day return.
- **Require true session adjacency.** Forward labels and return diagnostics are valid only when consecutive observations match the source's market-date calendar.
- **Compute time-series features on full ticker histories** before selecting the day's investable universe.
- **Lag the liquidity statistic** used to choose the top 1,000 names; avoid using future liquidity to select the sample.
- **Separate feature and target treatment.** Feature returns are clipped to **±50%** after quality checks; valid realized returns used for targets and P&L are **not** clipped.
- **Do not impute missing feature histories.** Apply minimum lookback requirements and track missing realized returns explicitly.

The notebook also audits security types, suspicious split-like moves, stale prices, coverage gaps, and the changing depth of the liquidity-ranked universe. These protections reduce avoidable errors but **do not** make a free vendor panel survivorship-safe.

## Signal construction

### 1. Intermediate-term momentum

A 12–1-style signal using the **sum of 231 market-demeaned daily simple feature returns**, shifted back by **21 sessions**. The formation window has a minimum of 120 valid returns.

This is a *sum-of-returns proxy*, **not** an exactly compounded 12–1 total return. The skipped month avoids mixing longer-horizon continuation with the most immediate reversal effect.

### 2. Short-horizon reversal

The negative average of cumulative feature returns over **3-, 5-, and 10-session** windows, each shifted back one session. The one-session skip ensures the reversal feature uses returns only through **T−1**, rather than relying on the same day's price move.

### 3. Daily cross-sectional ranking

Each raw signal is independently ranked within the daily eligible universe, transformed to approximately **[−1, 1]**, and combined without fitted weights:

$$
S_{i,t} = 0.5\,R_t(\mathrm{Momentum}_{i,t}) + 0.5\,R_t(\mathrm{Reversal}_{i,t}).
$$

Subtracting the next day's cross-sectional mean from the label does not change stock rankings, but makes the relative-return interpretation explicit.

## Modeling and evaluation

The baseline is tested against an incrementally more complex set of alternatives:

| Model | Inputs | Purpose | Matched raw OOS IC |
|---|---|---|---:|
| Equal-weight rank blend | Momentum + reversal | Interpretable, unfitted benchmark | **0.0124** |
| Core Ridge | Momentum + reversal | Test learned linear weights | **0.0124** |
| Conditioned Ridge | Core + volatility, prior absolute return, liquidity/size proxy | Test state-dependent information | **0.0159** |
| Gradient boosting | Same conditioned feature set | Test nonlinear effects and interactions | **0.0173** |

Models use **annual expanding-window refits**, an initial **three-year training period**, and comparisons on **identical historical evaluation dates**. The primary metric is per-date Spearman rank information coefficient (IC); paired **21-session block-bootstrap** intervals quantify uncertainty.

**Important distinction:** The GBM's raw IC improvement over the blend is approximately **+0.00491**, with a 95% block-bootstrap interval **[+0.00128, +0.00854]**. However, after removing joint volatility/size/prior-move exposures as an **attribution diagnostic**, the incremental gain shrinks to approximately **+0.0010** with an interval crossing zero. Neutralization is used to understand *why* the model works, not silently substituted for the raw investable prediction.

Additional feature tests examine gap fade, intraday fade, close location, volatility term structure, low volatility, MAX, and skewness. **Low-volatility and MAX signals retain incremental information** after testing multiple candidates, but they are strongly related and are treated as one signal family rather than independent discoveries.

## From prediction to portfolio performance

Rank IC measures the quality of ordering across the **entire** universe. A tradable portfolio instead cares about the **tails**, position sizing, trading frequency, and implementation costs.

The notebook tests a long-short decile portfolio with inverse-volatility weights:

- Long the highest-ranked decile; short the lowest-ranked decile.
- Scale each side by inverse recent volatility and target **50% long / 50% short**, for unit gross exposure.
- Evaluate gross P&L, annualized Sharpe, maximum drawdown, position turnover, and break-even transaction-cost budgets.
- Compare predictions on matched evaluation windows and explicitly stress the effect of missing realized returns.

A stronger ML model **does not improve the tested portfolio**: its matched-window gross Sharpe is **0.141**, versus **0.566** for the equal-weight rank blend. The mismatch motivates further investigation of tail ranking, exposures, risk sizing, and execution rather than simply adding more model complexity.

### Overnight concentration and execution

The same daily signal predicts overnight returns much better than intraday returns:

| Signal | Close → next open IC | Next open → close IC |
|---|---:|---:|
| Momentum | **0.0350** | 0.0013 |
| Reversal | **0.0137** | −0.0003 |
| Equal blend | **0.0344** | −0.0001 |

But a close→open strategy that **fully enters and exits each day** has roughly **2.0 units of turnover/day**, compared with **0.566** for the continuously held close→close book. Its **1.83 bps** break-even cost budget is correspondingly tight; a stylized **2 bps per dollar traded** assumption produces a **−0.143 net Sharpe** in that overnight simulation.

These are **reference-price simulations**, not executable auction backtests. Spread, impact, borrow, slippage, and capacity are not modeled. In addition, using a fully observed day-T OHLCV bar to target a trade at that same close requires further execution-timing validation.

## Robustness, interpretation, and limitations

The research includes tests that could falsify an appealing story rather than only amplify it:

- **Market regimes and selection:** the stronger recent IC is sensitive to universe composition. The recent-minus-early IC difference is **+0.0074** for the top 1,000 but only **+0.0006** for the top 250 names.
- **Reversal persistence:** a one-session-skipped reversal has IC **0.0070**; skipping a second session gives **0.0068**. This argues against the entire signal being a same-close artifact, without ruling out microstructure effects.
- **Conditional pressure:** reversal is stronger after high abnormal volume (high-minus-low IC **+0.0082**) and in past-only high-dispersion states (**+0.0105**). These are associations, **not causal identification**.
- **Portfolio fragility:** missing-return stress, turnover conventions, drawdowns, and break-even cost sensitivity are evaluated separately from rank prediction.

**Principal unresolved risks:**

1. **Survivorship and security identity:** the free panel does not reliably restore delisted companies or a point-in-time common-stock security master.
2. **Data conventions:** corporate actions, dividend treatment of open prices, and missing source-market sessions require independent verification.
3. **Historical selection:** the research inspected recent periods during development; it has **no untouched holdout**, and expanding fits do not explicitly purge all boundary labels.
4. **Risk exposures:** market demeaning is not the same as beta, sector, size, or comprehensive factor neutrality.
5. **Execution realism:** close/open tradability, auction liquidity, market impact, borrowing, and capacity need dedicated data and models.

**Next decisive experiment:** freeze the signal specification and replicate it on an independent **survivorship-safe historical common-stock dataset** with verified sessions and corporate actions. The precommitted replication gate is **mean IC > 0.005 with a 95% block-bootstrap confidence interval entirely above zero**. A failure is evidence against advancing that specification, not an invitation to retune on the replication set.

## Repository contents

```text
.
├── README.md                            # Project overview and verified research summary
├── Cross-Section-Equity-Research.ipynb  # Full pipeline, plots, ablations, diagnostics, backtests
└── Project-Research.pdf                 # Research narrative and presentation
```

`panel.parquet`, the large Stooq-derived input, **is not included** in the supplied project files. Its assembly script and original download manifest are also not embedded in the notebook. Consequently, the notebook contains recorded outputs but is **not reproducible from this repository alone**.

## How to run

The notebook was developed in **Google Colab (Python 3.11)** and mounts Google Drive. A sufficiently memory-capable runtime is recommended given the approximately 20.6-million-row source panel.

1. Obtain or reconstruct the Stooq-derived `panel.parquet` with the columns expected by the notebook (including `ticker`, `date`, `open`, `high`, `low`, `close`, `vol`). Validate its security mapping and corporate-action conventions independently.
2. In Colab, make sure the required packages are available:

   ```python
   %pip install pandas numpy scipy matplotlib scikit-learn pyarrow lightgbm
   ```

3. Put `panel.parquet` in your own Drive folder. In the **Setup** cell, update:

   ```python
   DRIVE = "/content/drive/MyDrive/NYSE stocks data"
   ```

4. Run the notebook **top to bottom**. It writes intermediate stages (`panel_continuous`, `panel_clean`, `panel_feat`, `panel_live`) under a run-specific subdirectory with simple provenance manifests.
5. Check **§11, Appendix — reproducibility check**. The supplied notebook's last recorded run reports that all quoted headline numbers match that run and that stages were generated in order.

The notebook can fall back to scikit-learn's histogram gradient boosting if LightGBM is unavailable; results may differ between model implementations. Reproducing the exact reported figures also requires the original input panel and compatible library versions.

## Tools and techniques

`Python` · `pandas` · `NumPy` · `SciPy` · `scikit-learn` · `LightGBM` · `Matplotlib` · `Parquet` · `cross-sectional factor research` · `walk-forward validation` · `bootstrap inference` · `portfolio backtesting` · `transaction-cost analysis`

---

**Research by Rishit Belani.** This repository documents exploratory quantitative research for educational and analytical purposes. The historical backtests are not investment advice, verified live results, or evidence of a deployable strategy.
