# IL&FS Default (2018): Black–Scholes Pricing, Implied Volatility, Greeks & Delta Hedging

### A quantitative event study of NIFTY and BANKNIFTY index options

This project examines how the September 2018 IL&FS default coincided with changes in option-pricing errors, implied volatility, option Greeks, and delta-hedging performance in India's index-options market.

The analysis compares NIFTY, representing the broader market, with BANKNIFTY, which has concentrated exposure to the financial sector.

## Research Question

**Did the IL&FS default coincide with a structural shift in option pricing and hedging performance?**

The study compares observations before and after 4 September 2018, examining whether the responses differed between NIFTY and BANKNIFTY.

## Study Overview

| Item | Description |
|---|---|
| Event | IL&FS default, 4 September 2018 |
| Sample period | June–December 2018 |
| Markets | NIFTY and BANKNIFTY index options |
| Observations | 248,227 |
| Pricing framework | Black–Scholes |
| Main focus | Pricing accuracy, implied volatility, Greeks and delta hedging |

## Research Objectives

- Evaluate changes in Black–Scholes pricing errors around the event.
- Examine changes in implied volatility and option Greeks.
- Assess delta-hedging errors and hedge effectiveness.
- Compare the response of NIFTY with that of BANKNIFTY.
- Test whether the observed pre/post differences are statistically significant.

## Methodology

The project combines option-pricing analysis, statistical testing and regression analysis.

1. **Data preparation and validation** — prepare and validate historical spot, derivatives, volatility and risk-free-rate inputs.
2. **Option valuation** — calculate Black–Scholes prices and associated option Greeks.
3. **Implied volatility** — examine implied-volatility behaviour around the event.
4. **Pricing diagnostics** — compare model prices with observed market prices and analyse pricing errors.
5. **Delta hedging** — evaluate hedging errors and variance-based hedge effectiveness.
6. **Statistical inference** — use Welch's t-tests, Mann–Whitney U tests, Levene tests and HC3-robust OLS regression to investigate differences and associations.

## Key Findings

The accompanying research report documents the following findings:

- **Pricing errors increased:** mean absolute pricing error rose from 36.70 in the pre-event sample to 52.85 in the post-event sample.
- **Implied volatility increased:** the report finds significant increases in both NIFTY and BANKNIFTY.
- **The response differed across indices:** BANKNIFTY exhibited a materially larger incremental response in pricing error, implied volatility and hedging error.
- **Hedging precision deteriorated:** absolute delta-hedging errors increased after the event.
- **Risk reduction remained meaningful:** variance-based hedge effectiveness remained above 0.90 across most sub-periods and improved for NIFTY.

These findings suggest that the post-event deterioration was not uniform across the two indices. The results do not, by themselves, establish that the IL&FS default caused every observed change.

## Repository Structure

```text
ilfs-index-options-analysis/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   ├── 01_Data_Download_ILFS.ipynb
│   ├── 02_Data_Cleaning_ILFS.ipynb
│   └── ilfs_option_pricing_hedging_analysis.ipynb
├── src/
│   └── 03_Data_Cleaning_Descriptive_Statistics_ILFS.py
├── data/
│   └── README.md
├── figures/
│   └── Research visualisations
└── tables/
    └── Statistical and performance summaries
```

The notebook and script filenames above reflect the intended organisation; consult the repository for the exact current filenames.

## Technologies

- Python
- Pandas and NumPy
- SciPy
- Statsmodels
- Matplotlib and Seaborn
- Jupyter Notebook

## Reproducibility

The repository contains the analysis notebooks, Python scripts, figures, tables and dependency list.

The full market-data files are not included. Reproduction therefore requires access to the relevant historical data and preparation of the input files expected by the notebooks and scripts. See [`data/README.md`](data/README.md) for data requirements.

Install the listed dependencies with:

```bash
pip install -r requirements.txt
```

Review the notebook execution order and input paths before running the analysis in a new environment.

## Limitations

- A before/after event study cannot independently establish causality.
- Results may be affected by other market developments during the sample period.
- Black–Scholes relies on assumptions that may not hold during stressed markets.
- Conclusions depend on data quality, model inputs and the chosen event window.
- Historical results are not a guarantee of future pricing or hedging performance.

## Data Source

Historical market data were obtained from the National Stock Exchange of India (NSE). Redistribution and reuse remain subject to the applicable data-use terms.

## Disclaimer

This repository is for academic research and educational purposes. It does not constitute investment advice or a recommendation to trade any financial instrument.
