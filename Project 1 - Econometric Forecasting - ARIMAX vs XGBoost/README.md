# Project 1: Econometric Forecasting: ARIMAX vs XGBoost
A comparative analysis of ARIMAX and gradient-boosted tree models for macro-financial forecasting. The study evaluates:
 - Out-of-sample predictive performance
 - Parameter stability and interpretability
 - Suitability of models for long-horizon forecasts
The objective is to assess trade-offs between structural econometric models and flexible machine-learning approaches in macroeconomic contexts.

-----

## Compute Environment and Performance Design
This project primarily uses CPU-based workflows with NumPy and Pandas. However, the XGBoost model has been implemented in a CUDA-enabled WSL environment to boost model training and test times.

-----

## Data Sourcing and Economic Rationale
1. Database on Indian Economy (DBIE): CPI inflation proxies and USD/INR exchange rates
2. Ministry of Statistics and Programme Implementation (MoSPI): Official CPI inflation data
3. Federal Reserve Bank of St. Louis (FRED): Brent crude oil prices
4. Masterstroke Online: RBI repo rate history

Data selection is driven by economic relevance, data integrity, and alignment with statistical tests that demonstrate relationships between exogenous (Repo, FX, CFPI and Brent) and endogenous (CPI Inflation) variables.

-----

## Pipelines
### ARIMAX Pipeline
The pipeline for implementation of any classical time series econometric model starts with statistical diagnostics on the underlying series to detect trend-stationarity, seasonality, and autocorrelation patterns. These tests are critical in determining the values for model coefficients (p, d, q, P, D, Q, s) which are explained in the subsequent sections. Hence, our implementation employed the following steps:

*1. Stationarity and Seasonality Tests:*

ARIMA-family models require the input series to be stationary. Therefore, statistical tests are performed to determine whether trend differencing (d) or seasonal differencing (D) is required. This is because the mathematics of such models rely on the underlying data to be stationary (both from trend and seasonality perspectives). Hence, tests such as ADF (Augmented Dickey Fuller) and KPSS (Kwiatkowski-Phillips-Schmidt-Shin) are employed to determine the need for trend differencing and Hyndman and CH (Canova-Hansen) are employed to determine the need for seasonality differencing. Additionally, an ACF (Autocorrelation Function) plot is developed to determine persistence (how well the past captures the present) and repeating autocorrelation patterns in the underlying data. This notebook performs all these tests and explains the underlying math as well as an interpretation of each of these tests (in the context of our use case).

*2. Model Implementation and Tuning:*

Based on identified parameters for trend and seasonality differencing, the model is then implemented on the dataset (in conjunction with the exogenous data) and fine-tuned based on its performance on in-sample-statistics such as log-likelihood, AIC (Akaike Information Criterion), BIC (Bayesian Information Criterion) and p-values of the various model coefficients. Additionally, residual diagnostics (including Ljung–Box tests for autocorrelation) were also examined to validate model adequacy and a train-test split of 80-20 was employed. Although multiple iterations of model parameters were employed to identify the model with best in-sample-performance, for simplicity purposes, this notebook does not provide a comparison of these iterations.

### XGBoost Pipeline
Since XGBoost is a supervised machine learning algorithm with no inherent notion of temporal dependence, lagged features—applied to both the target (endogenous variable) and the exogenous variables—must be manually engineered to capture autoregressive structure. Additionally, to improve model training time, the XGBoost was implemented in a CUDA (Compute Unified Device Architecture)-enabled WSL (Windows Subsystem for Linux) environment. Hence, the implementation employed the following steps:

*1. Feature Engineering:*

For our use case, we employed the development of the following features (on both the target as well as available features):

 - Lags (1, 2 and 3)
 - Moving Averages (3, 6 and 12)
 - Rolling Volatility (3, 6 and 12)
 - Month-on-Month %age Changes
 - First-Order Differencing
 - Month, Quarter and Year splits of the dates

*2. Model Training, Test and Hyperparameter Tuning:*

Subsequent to feature engineering, the data was split into train (90%) and test (10%). While the train and test split goes against established conventions of 80-20, the split of 90-10 was identified since ML models require more data than ARIMA during training and a 10% test set is still statistically meaningful in time-series forecasting. Additionally, hyperparameters were tuned using iterative search over learning rate, max_depth, subsample, column subsampling rate and regularization parameters (L1 and L2). Similar to the approach adopted for ARIMAX above, while multiple iterations of model parameters were employed to identify the best-fit model, for simplicity purposes, this notebook does not provide a comparison of these iterations.

-----

## Summary of Findings
While XGBoost outperforms ARIMAX on out-of-sample accuracy metrics, its lack of statistical uncertainty quantification makes long-horizon macro-economic interpretation more challenging. ARIMAX, on the other hand, produces structurally consistent forecasts that align with economic theory and provide interpretable uncertainty bounds.
XGBoost’s performance should also be viewed relative to dataset size. With only ~140 monthly observations, the dataset is small for a machine-learning model that typically benefits from hundreds or thousands of samples to learn complex nonlinear structures. Despite this limitation, XGBoost still demonstrates superior short-term predictive power.
Ultimately, the choice between ARIMAX and XGBoost depends on the objective of the analysis:

 - If the goal is point forecast accuracy in the short term → XGBoost is preferred.
 - If the goal is macro-economic plausibility, interpretability, and structured long-run forecasting → ARIMAX is preferred.

Thus, model selection is driven not only by performance metrics but also by dataset size, interpretability requirements, structural consistency, and the intended forecasting horizon.