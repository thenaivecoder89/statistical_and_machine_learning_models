# Project 2: Risk-Aware Portfolio Construction — Monte-Carlo vs Machine Learning–Based Risk Models

A comparative study of parametric Monte-Carlo, non-parametric (bootstrap) Monte-Carlo, and machine-learning–based risk estimation frameworks for risk-adjusted portfolio construction. The study evaluates how alternative risk-modeling assumptions propagate into downstream portfolio behavior, with specific emphasis on:

- Conditional risk estimation under non-Gaussian return dynamics
- Tail-risk behavior and downside protection
- Portfolio stability, turnover dynamics, and regime sensitivity
- Out-of-sample robustness across market stress periods

The objective is to assess the trade-offs between distributional assumptions, empirical resampling techniques, and data-driven machine learning models when used as inputs to a fixed, risk-controlled portfolio construction framework.

-----

## Compute Environment and Performance Design

This project relies primarily on Python-based analytical workflows using NumPy, Pandas, SciPy, and Scikit-learn–compatible tooling for statistical diagnostics and Monte-Carlo simulation. Machine-learning components are implemented using XGBoost, with CUDA acceleration leveraged where available to improve training efficiency and enable extensive rolling out-of-sample backtesting.

Given the simulation-heavy nature of Monte-Carlo–based risk modeling and the multi-year rolling evaluation framework, the implementation emphasizes reproducibility, numerical stability, and computational efficiency over brute-force optimization.

-----

## Data Sourcing and Market Rationale

The analysis is conducted on a multi-asset portfolio constructed from liquid, representative market instruments spanning equity and defensive assets. Asset selection is motivated by diversification objectives, availability of long historical return series, and relevance for institutional-style portfolio construction.

Data preprocessing is designed to ensure:
- Alignment of return frequencies across assets
- Consistent treatment of missing data and outliers
- Preservation of empirical distributional properties for non-parametric modeling

Upfront statistical diagnostics confirm the presence of skewness, excess kurtosis, volatility clustering, and time-varying correlations, motivating the explicit comparison between Gaussian assumptions, empirical resampling, and conditional ML-based risk estimation.

-----

## Modeling Pipelines

All approaches are embedded within an identical portfolio construction rule, ensuring that observed differences in outcomes are attributable solely to the risk model, not to changes in optimization logic.

### Gaussian Monte-Carlo Pipeline

The parametric Monte-Carlo framework assumes multivariate normality of asset returns, with mean vectors and covariance matrices estimated over rolling historical windows. Simulated return paths are generated using these parameters and fed into the portfolio construction engine to derive risk-adjusted allocations.

This approach serves as a classical benchmark, offering analytical tractability and interpretability, but relies heavily on the validity of normality assumptions.

### Bootstrap (Non-Parametric) Monte-Carlo Pipeline

The bootstrap Monte-Carlo framework replaces parametric assumptions with empirical resampling of historical return vectors. By preserving observed marginal distributions, dependence structures, skewness, and fat tails, this approach provides a distribution-aware alternative to Gaussian simulation.

The bootstrap framework is particularly suited for capturing tail behavior and stress-period dynamics, at the cost of reduced extrapolative power beyond observed regimes.

### XGBoost-Based Risk Modeling Pipeline

The machine-learning approach employs XGBoost models to estimate conditional volatility and correlation dynamics as functions of lagged returns and engineered features. Rather than simulating returns directly, the ML model produces time-varying risk estimates that adapt to evolving market conditions.

This framework prioritizes regime sensitivity and responsiveness, enabling rapid adjustment to changing volatility and correlation structures, but offers limited structural interpretability relative to classical Monte-Carlo approaches.

-----

## Evaluation Framework

All models are evaluated using a multi-year rolling out-of-sample backtesting framework, ensuring strict separation between estimation and evaluation periods. Performance is assessed using risk-centric metrics, including:

- Volatility control and realized risk alignment
- Tail loss frequency and drawdown behavior
- Portfolio turnover and allocation stability
- Risk-adjusted return measures

This design explicitly avoids return forecasting or alpha generation, focusing instead on risk estimation quality and downstream portfolio behavior.

-----

## Summary of Findings

The results demonstrate that risk-model choice plays a critical role in shaping portfolio outcomes, particularly during periods of market stress.

Bootstrap Monte-Carlo consistently delivers superior tail-risk control and portfolio stability by preserving empirical distributional characteristics observed in historical data. Gaussian Monte-Carlo remains a useful baseline but exhibits limitations when underlying return dynamics deviate from normality.

The XGBoost-based approach shows heightened regime sensitivity and conditional responsiveness, making it well-suited for risk regime identification and adaptive risk control, though less appropriate as a standalone portfolio optimization engine.

Overall, the findings highlight the importance of aligning risk-modeling methodology with underlying data characteristics and portfolio objectives, demonstrating that robust risk-aware portfolio construction depends as much on modeling assumptions as on optimization logic itself.