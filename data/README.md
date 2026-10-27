# Data Documentation

## Project Overview

This project examines the impact of the September 2018 IL&FS default on Black–Scholes option pricing, implied volatility, option Greeks, and delta-hedging performance in NIFTY and BANKNIFTY index options.

**Study period:** June–December 2018  
**Event date:** 4 September 2018

## Data Requirements

The analysis uses historical market data covering:

- NIFTY spot prices
- BANKNIFTY spot prices
- Indian VIX
- Risk-free interest rates
- NIFTY and BANKNIFTY derivatives and options

Refer to the data-loading sections of the notebooks and scripts for the expected filenames, column names, and formats.

## Data Availability

Raw and large processed datasets are not included in this repository.

The historical market data were sourced from the National Stock Exchange of India (NSE). Users seeking to reproduce the analysis should obtain the relevant data through authorised sources and comply with the applicable access, usage, and redistribution terms.

## Data Preparation

The project includes data-cleaning, validation, descriptive-statistics, option-pricing, and hedging-analysis workflows.

Before reproducing the analysis, check date formats, missing values, option types, strike prices, expiry dates, market prices, volatility inputs, and time-to-expiry conventions.

## Reproducibility

Install the Python dependencies listed in the repository's `requirements.txt` file.

Prepare the input datasets in the formats expected by the notebooks and scripts, then follow the analysis workflow documented in the repository.

Exact input paths and filenames may need to be adjusted for your local environment.
