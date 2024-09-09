---
title: typology of data drift
description: Writting about drift in data and what's the impact of it
date: 2024-09-09
categories:
  - MLOPs
  - ML
tags:
  - MLOps
  - data-drift
  - covariate-shift
  - prior-probability-shift
  - sample-selection-biais
  - model-observability
---



## Types of Drift in Machine Learning

### 1. Data Drift

**Notation :**

- $$P_{\text{train}}$$: Distribution of the training data
- $$P_{\text{test}}$$: Distribution of the test or production data
- $$X$$: Input variables (features)
- $$Y$$: Target variable (to be predicted)

$$C$$
: A confounding variable that affect X or Y 


**Mathematical Formulation :**

<figure><center><img src="https://i.imgur.com/VLKoYT5.png" alt=""></center></figure>


$$
P_{\text{train}}(X, Y) \neq P_{\text{test}}(X, Y)
$$

There's 3 main subcategories (covariate shift , Prior probability shift &  Sample Selection Bias:)

#### a) Covariate Shift:

Covariate shift occurs when the distribution of the input features (i.e., the characteristics or attributes) changes between the training and test data, but the way the features relate to the target (the thing you're trying to predict) remains the same.

<figure><center><img src="https://i.imgur.com/fg5GrSy.png" alt="" width="50%" ></center></figure>

$$P_{\text{train}}(X) \neq P_{\text{test}}(X) \ \text{, but} \ P_{\text{train}}(Y \mid X) = P_{\text{test}}(Y \mid X)$$


**SubSubcategories:**

**1. Covariate Observation Shift (COS)**
Occurs when the conditional distribution of covariates given the labels shifts between training and testing:

$$P(X \mid Y)_{\text{train}} \neq P(X \mid Y)_{\text{test}}$$

- **Example**: A model trained on animal images in a controlled environment tested on outdoor images with different lighting conditions.

**2. Covariate Hidden Shift (CHS)**
Involves the presence of a hidden variable $$X_H$$ that alters the covariates. The conditional distribution of hidden covariates given the observed covariates changes:

$$P(X_H \mid X)_{\text{train}} \neq P(X_H \mid X)_{\text{test}}$$

- **Example**: A sales prediction model trained without considering economic changes as a hidden variable.

**3. Distorted Covariate Shift (DCS)**
Occurs when observed covariates are noisy or distorted compared to the true covariates:

$$P_{\text{observed}}(X) \neq P_{\text{true}}(X)$$

- **Example**: Faulty sensors in a manufacturing process providing inaccurate data to a predictive model.



#### b) Prior Probability Shift:

Prior probability shift happens when the overall proportion of the different classes or categories in the target variable changes between training and test data, but the relationship between the features and the target stays consistent.

<figure><center><img src="https://i.imgur.com/BJWCzc4.png" alt="" width="50%" ></center></figure>

  $$P_{train}(Y) \neq P_{\text{test}}(Y)\ \text{, but } P_{\text{train}}(X \mid Y) = P_{\text{test}}(X \mid Y)$$ 
  
**1. Prior Probability Shift (PPS)**
This shift occurs when the prior probability distribution of the labels changes between the training and testing phases, without affecting the covariates:

$$
P(Y)_{\text{train}} \neq P(Y)_{\text{test}}
$$

- **Example**: A model trained on a dataset where one class is more frequent may face difficulty when tested on a dataset with different class proportions.

**2. Prior Probability Observation Shift (PPOS)**
Occurs when the prior probability distribution changes, and there is also an unobserved (hidden) factor $$X_H$$ that affects the covariates. The relationship between the hidden covariate and the label changes:

$$
P(X_H \mid Y)_{\text{train}} \neq P(X_H \mid Y)_{\text{test}}
$$

- **Example**: In a customer segmentation task, an unobserved variable like customer behavior changes over time, affecting the relationships between segments and their features.

**3. Prior Probability Hidden Shift (PPHS)**
This type of shift occurs when a hidden variable $$X_H$$ influences both the covariates and the labels, and its conditional distribution shifts between the training and test datasets:

$$
P(X_H \mid Y)_{\text{train}} \neq P(X_H \mid Y)_{\text{test}}
$$

- **Example**: A medical model trained on data where $$X_H$$ (e.g., certain health conditions) influences both the test results and the patient outcomes, but changes in how those conditions manifest over time cause a shift.

**4. Distorted Prior Probability Shift (DPPS)**
Occurs when the prior probability distribution shifts and the covariates are distorted. The observed covariates do not match the true covariates:

$$
P_{\text{observed}}(X) \neq P_{\text{true}}(X) \quad \text{and} \quad P(Y)_{\text{train}} \neq P(Y)_{\text{test}}
$$

- **Example**: A model trained with clean data might perform poorly when tested on noisy data, combined with a shift in label distributions. For instance, sensor errors distort the data, and class distributions also change.


  
#### c) Sample Selection Bias:

Sample selection bias occurs when the training data is not a good representation of the real-world population or the test data. This typically happens because the data used for training is collected in a biased or unrepresentative way.





<script type="text/javascript"
  async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">
</script>

<script type="text/x-mathjax-config">
  MathJax.Hub.Config({
    tex2jax: {
      inlineMath: [['$','$'], ['\\(','\\)']],
      processEscapes: true
    }
  });
</script>
