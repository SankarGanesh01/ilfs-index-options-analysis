# ilfs-index-options-analysis
Quantitative event study of the 2018 IL&amp;FS default: Black–Scholes pricing, implied volatility, option Greeks and delta hedging in NIFTY and BANKNIFTY options.
# Impact of the IL&FS Default (2018) on Black–Scholes Pricing Accuracy, Implied Volatility, Option Greeks, and Delta Hedging

**A quantitative event study of NIFTY and BANKNIFTY index options**

## Overview

This project investigates how the 2018 IL&FS credit crisis relates to option pricing, implied volatility, option sensitivities, and delta-hedging performance in India's index options market.

The analysis examines NIFTY and BANKNIFTY options around the IL&FS event, comparing pre-event and post-event behaviour using financial modelling, statistical tests, and regression analysis.

## Research Questions

- Did Black–Scholes pricing errors change following the IL&FS event?
- How did implied volatility and the distribution of option Greeks differ before and after the event?
- Did delta-hedging performance change across the two periods?
- Were the observed differences statistically significant, and did the responses differ between NIFTY and BANKNIFTY?

## Methodology

The research workflow includes:

- Options and underlying-market data preparation and validation
- Black–Scholes option pricing and implied-volatility analysis
- Examination of Delta, Gamma, Theta, Vega, and Rho
- Pre-event versus post-event comparisons
- Statistical testing and regression analysis
- Evaluation of delta-hedging performance

The implementation details, assumptions, sample definitions, and evaluation metrics will be documented alongside the analysis code.

## Markets and Event

- **Markets:** NIFTY and BANKNIFTY index options
- **Event:** IL&FS credit crisis in 2018
- **Event date used in the study:** 4 September 2018
- **Study window:** June–December 2018

## Repository Structure

The repository will contain the analysis notebooks, reusable code where applicable, selected figures and summary tables, and documentation describing the data requirements.

Large datasets will not be included by default. Data sources, access requirements, and reproduction instructions will be documented separately.

## Results and Interpretation

The findings will be presented with the relevant summary tables, statistical tests, and figures after the published outputs have been checked against the underlying code.

## Limitations

The analysis is an empirical study around a major market event. Pre- and post-event differences should not automatically be interpreted as causal effects of the IL&FS default. Results may also depend on the option sample, data quality, model assumptions, market conditions, and hedging conventions.

## Reproducibility

Instructions for installing dependencies, accessing the required data, and running the notebooks will be added after the project files and dependencies have been reviewed.

## Disclaimer

This repository is intended for academic and quantitative-finance research. It is not investment advice.
