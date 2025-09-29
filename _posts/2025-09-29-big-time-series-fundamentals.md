---
title: The 7 big time series fundamentals
description:
date: 2025-09-29
categories:
  - MLOPs
tags:
  - MLOps
  - Machine
  - Learning
  - System
  - MQTT
  - Mosquitto
  - Streaming
  - Real Time
pin: true
math: true
mermaid: true
image:
  path: /assets/img/panels/panel9@4x.png
---

## 0. **Heuristic Models (Baseline and Simple Methods)**

Before jumping to complex models, master the foundational heuristic approaches that often serve as surprisingly strong baselines. These include naive forecasting (last observed value), seasonal naive (last value from same season), moving averages, and exponential smoothing methods (Simple, Holt's, and Holt-Winters). 

## Rule-Based (EDA-driven)

We begin with the simplest setting: you have **few discrete features** and want a forecast based on **stable statistics** rather than stochastic assumptions.  
Using **boxplots** per group, you can test whether the **median** in a group $g$ is stable across time.  
If it is, you simply forecast that median $m_g$.  
If recent means are more stable, you use a moving mean instead.  
When clear seasonality dominates, you repeat the seasonal lag value; otherwise, you fall back to the last observation.

$$
\hat y_t =
\begin{cases}
m_g & \text{if median in group $g$ is stable across time} \\
\dfrac{1}{k}\sum_{i=1}^k y_{t-i} & \text{if recent mean passes stability tests} \\
y_{t-s} & \text{if seasonal lag is robust} \\
y_{t-1} & \text{fallback if no rule applies}
\end{cases}
$$

where  

$$
m_g = \mathrm{median}\{y_j : j \in g\}.
$$

This approach works surprisingly well when group behavior is consistent.  
But if a **new unseen group** appears, $c_t \notin \{\text{known rules}\}$, the function $f(c_t)$ is undefined, and the error  

$$
e_t = y_t - f(c_t)
$$  
can become arbitrarily large.  
So we move towards **stochastic models** beginning with the simplest one: the *Naive* model.

**Sanity check:**  
Residuals should show **no seasonal spikes** in the ACF if seasonality is now captured.



---

#### Seasonal Naive  

The **Naive model** assumes that the **best forecast for tomorrow** is simply **today’s observation**:

$$
\hat y_{t+1} = y_t
$$

This works if the data behaves like a **random walk** or changes very slowly.  
But when the true process has **seasonality**  

$$
y_t = \mu + s_t + \epsilon_t, \qquad s_t = s_{t-s},
$$

the one-step-ahead error becomes  

$$
e_{t+1}^{Naive} = (s_{t+1}-s_t)+(\epsilon_{t+1}-\epsilon_t),
$$

and the seasonal difference $(s_{t+1}-s_t)$ introduces bias whenever $s_{t+1} \neq s_t$.  

**Conclusion:** The Naive model is too simple → we use **Seasonal Naive** instead:

$$
\hat y_{t+h} = y_{t+h-s}
$$

Here, forecasts repeat the value from the **same point in the previous season**, removing seasonal bias.  
But this introduces a new problem: the noise from the previous season gets copied forward:

$$
\mathrm{Var}(e_{t+1}^{SN}) = 2\sigma^2
$$

So forecasts become very **noisy**, pushing us toward **noise reduction** via *Moving Averages*.


**Sanity check:**  
As   $k  $ grows, the plot looks smoother; residual variance drops roughly by   $1/k  $.

> **Tip:** With seasonality, use **seasonal smoothing** (e.g., average each month across years) to avoid blurring the seasonal pattern.

---

#### Moving Average  

The **Moving Average** approach smooths the noise by taking the mean of the last $k$ points:

$$
\hat y_{t+1} = \dfrac{1}{k}\sum_{i=0}^{k-1} y_{t-i}
$$

As $k$ increases, the noise variance drops  

$$
\mathrm{Var}(\hat y) = \frac{\sigma^2}{k}
$$  

making forecasts **smoother** and less volatile.  

But if the true data has a **linear trend** $y_t = a + bt + \epsilon_t$, the moving average introduces a bias  

$$
\text{Bias} = b \cdot \frac{k+1}{2}
$$  

because the average lags behind the growing trend.  
Hence, we move to a model that can **adapt to level and trend**: *Simple Exponential Smoothing*.

**Sanity check:**  
As $\alpha\uparrow$, forecasts react faster; lag after jumps decreases.

> **Note:** SES produces a **flat multi-step forecast**: $\hat{y}_{t+h}=\hat{y}_{t+1}$ for all  $h$. That’s the next weakness.


---

#### Simple Exponential Smoothing (SES)

SES assigns **more weight to recent points**:  

$$
\hat y_{t+1} = \alpha y_t + (1-\alpha)\hat y_t, \qquad 0<\alpha<1
$$

Here $\alpha$ controls memory:  
- Large $\alpha$ → reacts quickly to changes  
- Small $\alpha$ → produces smoother forecasts  

But for a true linear trend  $$y_t = a+bt+\epsilon_t $$, SES h-step forecast is constant:
 $$
\hat{y}_{t+h}^{SES} = \hat{y}_{t+1}.
 $$
True grows by  $$bh$$. So the horizon-h bias:
 $$
\mathbb{E}[\hat{y}_{t+h}^{SES} - y_{t+h}] \approx -bh
 $$
→ linearly worsening under-forecast when  $$b>0 $$.
so it **cannot follow trends**, leading to *Holt’s Linear Trend* model.

---

#### Holt (Linear Trend)

Holt adds a **trend component** $T_t$ alongside the **level** $L_t$:

$$
\begin{aligned}
L_t &= \alpha y_t + (1-\alpha)(L_{t-1}+T_{t-1}) \\
T_t &= \beta(L_t-L_{t-1})+(1-\beta)T_{t-1} \\
\hat y_{t+h} &= L_t + hT_t
\end{aligned}
$$

Now the forecast grows along the slope $T_t$, eliminating trend bias.  
But seasonal effects remain in the residual  

$$
e_{t+1}^{Holt} = s_{t+1} + \epsilon_{t+1},
$$  

so we finally add **seasonality** in *Holt–Winters*.

**Sanity check:**  
Residuals shouldn’t show **systematic drift**; trend ACF should flatten.

> **Option:** *Damped trend*: multiply  $$T_t $$ by  $$\phi^h, 0<\phi<1 $$ when long-run growth shouldn’t explode.


---

####  Holt–Winters  model

With **Holt–Winters**, we model  
- **level** $L_t$,  
- **trend** $T_t$,  
- and **seasonality** $S_t$:  

Additive version:  

$$
\begin{aligned}
L_t &= \alpha(y_t - S_{t-s})+(1-\alpha)(L_{t-1}+T_{t-1})\\
T_t &= \beta(L_t-L_{t-1})+(1-\beta)T_{t-1}\\
S_t &= \gamma(y_t-L_t)+(1-\gamma)S_{t-s}\\
\hat y_{t+h} &= L_t + hT_t + S_{t+h-s}
\end{aligned}
$$

Multiplicative version:  

$$
\hat y_{t+h} = (L_t + hT_t)\times S_{t+h-s}
$$

Now the model captures **all three components**:  
- Level  
- Trend  
- Seasonality  

The choice between additive and multiplicative depends on whether seasonal amplitude grows with the level.Understanding why these simple methods work helps you appreciate when complexity is unnecessary and provides benchmarks that sophisticated models must beat. These methods also teach you about the fundamental components of time series: level, trend, and seasonality.

## 1. **Stationarity and Non-Stationarity**

Understanding the statistical properties of time series is crucial. A stationary series has constant mean, variance, and autocovariance over time. Most forecasting methods assume stationarity, so you need to master differencing, detrending, and transformation techniques. Learn to identify unit roots using tests like ADF (Augmented Dickey-Fuller) and KPSS, and understand the implications of working with non-stationary data.

## 2. **Autocorrelation and Partial Autocorrelation**

These concepts reveal the relationship between observations at different time lags. ACF (Autocorrelation Function) shows correlation between a value and its lags, while PACF (Partial Autocorrelation Function) shows the direct relationship after removing indirect effects. Mastering these helps you identify appropriate model orders for ARIMA models and understand the underlying temporal dependencies in your data.

## 3. **State Space Models and Kalman Filtering**

This provides a unified framework for understanding many time series models. State space representation allows you to model both observed and hidden (latent) states in a system. The Kalman filter is optimal for linear Gaussian models and forms the basis for understanding more complex methods like particle filters and sequential Monte Carlo methods. This is essential for real-time forecasting and handling missing data.

## 4. **Spectral Analysis and Frequency Domain Methods**

While most time series work happens in the time domain, spectral analysis transforms data into the frequency domain using techniques like Fourier transforms and periodograms. This is invaluable for identifying cyclical patterns, seasonal components, and hidden periodicities. Understanding wavelet transforms also allows you to analyze non-stationary signals with time-varying frequencies.

## 5. **Multivariate Time Series and Causality**

Move beyond univariate analysis to understand relationships between multiple time series. Master Vector Autoregression (VAR), cointegration (long-run equilibrium relationships), and Granger causality testing. These concepts are critical for understanding how different time series influence each other and for building systems of equations that capture complex interdependencies in economic, financial, or physical systems.

## 6. **Ergodicity, Ensemble vs Time Averages, and Non-Ergodic Processes**

This is a profound concept that challenges fundamental assumptions in time series analysis. Ergodicity means that time averages equal ensemble averages—that studying one realization over time gives you the same information as studying many realizations at one point in time. Most statistical methods assume ergodicity, but many real-world processes (financial markets, evolutionary systems, path-dependent phenomena) are non-ergodic. In non-ergodic systems, past observations may not be representative of future possibilities, individual trajectories diverge dramatically, and the concept of a stable "true" parameter becomes questionable. Understanding this helps you recognize when traditional forecasting breaks down, why backtesting can be misleading, and when you need robust decision-making frameworks rather than point forecasts. This concept bridges statistics, physics, and philosophy, fundamentally changing how you think about uncertainty and prediction.