---
title: drift-in-machine-learning
description: Writting about drift or shift in ml , and what's the impact of it 
date: 2024-09-09
categories:
  - MLOPs
  - ML
tags:
  - MLOps
  - data-drift
  - covariate-drift
  - concept-drift
  - model-observability
---



### Types of Drift in Machine Learning

**1. Data Drift**

**Notation:**

- $P_{\text{train}}$: Distribution of the training data
- $P_{\text{test}}$: Distribution of the test or production data
- $X$: Input variables (features)
- $Y$: Target variable (to be predicted)
- $P(X)$: Marginal distribution of the input variables
- $P(Y \mid X)$: Conditional distribution of $Y$ given $X$

**Mathematical formulation:**  
$P_{\text{train}}(X, Y) \neq P_{\text{test}}(X, Y)$ [Moreno-Torres et al., 2012]

**Subcategories:**

- **a) Covariate Shift:**  
  $P_{\text{train}}(X) \neq P_{\text{test}}(X)$, but $P_{\text{train}}(Y \mid X) = P_{\text{test}}(Y \mid X)$ [Shimodaira, 2000]
  
- **b) Prior Probability Shift:**  
  $P_{train}(Y) \neq P_{\text{test}}(Y)$, but $P_{\text{train}}(X \mid Y) = P_{\text{test}}(X \mid Y)$ [Moreno-Torres et al., 2012]
  
- **c) Sample Selection Bias:**  
  The training data is not representative of the target population [Quionero-Candela et al., 2009]

---

**2. Prediction Drift (Concept Drift)**

**Mathematical formulation:**  
$P_{t1}(Y \mid X) \neq P_{t2}(Y \mid X)$, where $t1$ and $t2$ represent different points in time [Gama et al., 2014]

**Subcategories:** [Widmer and Kubat, 1996]

- **a) Sudden:** Abrupt and sudden change
- **b) Gradual:** Progressive change over time
- **c) Incremental:** Small continuous changes
- **d) Recurrent:** Concepts change but can return to previous states
- **e) Temporary:** Temporary change that quickly returns to the original state

<figure><center><img src="https://raw.githubusercontent.com/IbLahlou/my-files/main/Concept_Drift_subtypes.png?token=GHSAT0AAAAAACWQZXSCFZNTNO2KRDLFH76EZW6J3IQ" width="75%" alt=""></center><center><em><figcaption>Concept Drift Subcategories</figcaption></em></center></figure>



---

**3. Covariate Drift (a specific type of data drift)**

**Mathematical formulation:**  
$P_{\text{train}}(X) \neq P_{\text{test}}(X)$, but $P_{\text{train}}(Y \mid X) = P_{\text{test}}(Y \mid X)$

**Subcategories:** [Moreno-Torres et al., 2012]

- **a) Feature Drift:** Changes in the distribution of one or more features
- **b) Domain Shift:** Changes in the feature space (e.g., new categories)
- **c) Magnitude Shift:** Changes in the scale or magnitude of the features

---

**4. Performance and Stability Tests**

Performance and stability tests are crucial for ensuring the robustness and reliability of machine learning models. They help detect potential drift and performance degradation, identify the causes of reduced performance, and ensure that the model remains performant and stable over time.

---

### Kolmogorov-Smirnov (KS) Test for Drift Detection

The Kolmogorov-Smirnov (KS) test is used to compare the distribution of features between the training and test data.


<figure><center><img src="https://raw.githubusercontent.com/IbLahlou/my-files/main/Kolmogrov-smirnov_test.png?token=GHSAT0AAAAAACWQZXSD5Y6MTNL3R3NOQ73YZW6J4JA" width="75%" alt=""></center><center><em><figcaption>CDF Graphs to calculate KS test metric</figcaption></em></center></figure>



```python
from scipy.stats import ks_2samp

# Example of KS test
ks_stat, p_value = ks_2samp(X_train[:, 0], X_test[:, 0])
print(f"KS Statistic: {ks_stat}, P-value: {p_value}")
```

Here is the translation of the provided text into English:

---

The formula for the KS test is given by: 

$$D_{n,m} = \sup_x |F_n(x) - F_m(x)|$$

where \( F_n(x) \) and \( F_m(x) \) represent the empirical cumulative distribution functions (CDFs) of the two respective samples. \\



If the p-value is low (e.g., less than 0.05), this indicates a significant drift in the feature distribution.

---

### Classifier Performance Degradation Test

Monitor performance degradation using common metrics such as accuracy, precision, or F1 score:

```python
from sklearn.metrics import accuracy_score

# Train a model and make predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Calculate accuracy
accuracy_train = accuracy_score(y_train, y_pred_train)
accuracy_test = accuracy_score(y_test, y_pred_test)

print(f"Training Accuracy: {accuracy_train}, Test Accuracy: {accuracy_test}")
```

A significant drop in accuracy can signal concept drift or data drift.

---

### Population Stability Index (PSI)

The PSI is often used in credit scoring to compare distributions between two datasets. It's calculated as follows:

The formula for PSI is:

$$\text{PSI} = \sum_{i=1}^{k} (O_i - E_i) \log\left(\frac{O_i}{E_i}\right)$$

where \( O_i \) and \( E_i \) refer to the observed and expected frequencies in segment \( i \), respectively. \\


```python
def calculate_psi(expected, actual, buckettype='bins', buckets=10):
    """Calculate the PSI (population stability index) for a single feature."""
    def scale_range(input, min, max):
        input += -(np.min(input))
        input /= np.max(input) / (max - min)
        input += min
        return input
    
    breakpoints = np.linspace(0, 1, buckets + 1)
    if buckettype == 'bins':
        expected_perc = np.histogram(expected, bins=buckets)[0] / len(expected)
        actual_perc = np.histogram(actual, bins=buckets)[0] / len(actual)
    else:
        expected_perc = np.percentile(expected, breakpoints)
        actual_perc = np.percentile(actual, breakpoints)
    
    psi = np.sum((actual_perc - expected_perc) * np.log(actual_perc / expected_perc))
    return psi

psi_value = calculate_psi(X_train[:, 0], X_test[:, 0])
print(f"PSI for Feature 0: {psi_value}")
```

PSI values:
- < 0.1: No significant change
- 0.1 - 0.2: Some drift
- > 0.2: Significant drift

---

### Drift Detection using Evidently

You can use monitoring libraries such as **Evidently** for detailed drift reports:

```python
from evidently.report import Report
from evidently.metrics import DataDriftMetric

# Initialize report
drift_report = Report(metrics=[DataDriftMetric()])
drift_report.run(reference_data=X_train, current_data=X_test)

# Show report
drift_report.show()
```

This will generate a visual drift report to help in understanding feature or concept drift across datasets.

---

### Drift Detection using NannyML

NannyML is another tool for drift detection and performance estimation. Below is an example of how to use it to detect data drift:

```python
import nannyml as nml

# Load reference and analysis datasets
reference_data = X_train.copy()
analysis_data = X_test.copy()

# Initialize a data drift calculator
drift_calculator = nml.DataDriftCalculator()

# Fit on reference data (training set)
drift_calculator.fit(reference_data)

# Run calculation on analysis data (test set)
results = drift_calculator.calculate(analysis_data)

# Visualize results
results.plot()
```

NannyML uses advanced techniques to estimate drift without access to the target variable in production, making it highly suitable for production monitoring where the ground truth is not available in real-time.
