---
title: Physics-Informed Feature Engineering
description: A comprehensive guide to energy forms, SI measurement standards, and practical feature engineering strategies for data scientists building physics-aware ML models with real-world battery discharge example
date: 2025-12-14
categories:
  - Physics
  - Data Science
  - Machine Learning
tags:
  - Energy Forms
  - SI Units
  - Measurement
  - Feature Engineering
  - Dimensional Analysis
  - Physics-Informed ML
  - Battery Modeling
pin: true
math: true
mermaid: true
reading_time: 18 min
image:
  path: /assets/img/panels/panel13@4x.png
---

# Forms of Energy

Energy is the ability to do work or cause change, and it appears in interchangeable forms. This overview lists the main categories with real-world examples, explains why each is considered a form of "energy," traces key historical discoveries, and notes the standard definitions. It also covers energy conservation, the role of SI units in measurement, and practical feature engineering guidance for data scientists working on modeling energy-related systems, such as in physics simulations, chemical reaction networks, or machine learning predictions.

## Main Categories

### Kinetic Energy
The energy associated with motion. Anything moving has kinetic energy, which can be transferred to other objects or converted to other forms.

- **Standard Definition**: Energy of motion (translational and rotational)
  $$
  KE_{\text{trans}} = \frac{1}{2} m v^2
  $$
  $$
  KE_{\text{rot}} = \frac{1}{2} I \omega^2
  $$
  Relativistic form:
  $$
  E = \gamma m c^2 - m c^2
  $$
- **Why It Is Called Energy**: It quantifies the capacity to perform work through motion (e.g., a moving object can push another). The concept unifies diverse phenomena under conservation laws, enabling predictive modeling in systems from particle physics to engineering.
- **History of Discovery**: Galileo (late 16th–early 17th century) studied falling bodies and inclined planes, laying groundwork. Huygens (1650s) formalized vis viva ($$ mv^2 $$) as a conserved quantity in collisions. Leibniz (1680s) named it and argued for $$ mv^2 $$ over momentum. The modern $$\frac{1}{2} mv^2$$ form emerged in the 19th century with Coriolis and the development of classical mechanics, influencing thermodynamics and statistical mechanics.
- **Subforms/Examples**:
  - **Thermal**: Random motion of particles (atoms/molecules). Example: Heat from friction or steam from boiling water. In data science, thermal energy datasets often involve temperature distributions for heat transfer models.
  - **Mechanical**: Macroscopic motion of objects. Example: Rolling ball; wind turning turbine blades. Useful in predictive maintenance models for machinery.
  - **Electrical**: Ordered motion of charged particles. Example: Current in household appliances. Key in energy consumption forecasting using time-series data.
  - **Sound**: Energy carried by vibrational waves in matter. Example: Guitar string or speaker cone vibrating. Analyzed in signal processing for acoustic modeling.

### Potential Energy
Stored energy that depends on the relative position or configuration of objects or particles, convertible to kinetic energy.

- **Standard Definition**: Energy stored by virtue of position in a force field or internal configuration
  $$
  PE_{\text{grav}} = m g h
  $$
- **Why It Is Called Energy**: It represents latent capacity to do work when released (e.g., falling object gains speed). Included because total mechanical energy (kinetic + potential) is conserved, simplifying simulations and optimizations in data-driven models.
- **History of Discovery**: Concept implicit in ancient water wheels and catapults. Formalized in the 19th century: Thomas Young coined "energy" in 1807; potential energy developed by Rankine, Kelvin, and others in the 1850s as part of the mechanical equivalent of heat and conservation laws, pivotal for the First Law of Thermodynamics.
- **Subforms/Examples**:
  - **Gravitational**: Due to position in a gravitational field. Example: Water behind a dam; book on a shelf. Modeled in geospatial data for hydropower predictions.
  - **Elastic**: Stored in deformed elastic materials. Example: Stretched rubber band, compressed spring, drawn bow. Used in finite element analysis datasets.
  - **Chemical**: Stored in molecular bonds and arrangements. Example: Gasoline, food, battery discharge. Critical for reaction kinetics modeling in cheminformatics.
  - **Nuclear**: Stored in atomic nuclei via strong force. Example: Fission in reactors; fusion in the Sun. Involved in radiation transport simulations.

### Radiant Energy (Electromagnetic)
Energy carried by electromagnetic waves, traveling through space at the speed of light.

- **Standard Definition**: Energy of electromagnetic radiation, quantified by photon energy or wave intensity
  $$
  E_{\text{photon}} = h \nu
  $$
  $$
  I = \frac{P}{A}
  $$
- **Why It Is Called Energy**: It can perform work (e.g., heat objects, drive photovoltaic cells) and is conserved/interconvertible with other forms, allowing for unified models in quantum and classical regimes.
- **History of Discovery**: Early observations of light/heat from the Sun. Maxwell (1860s) unified electricity and magnetism into electromagnetic waves. Planck (1900) introduced quanta to explain blackbody radiation; Einstein (1905) explained photoelectric effect with photons, solidifying radiant energy as discrete packets, foundational for quantum mechanics.
- **Examples**: Sunlight powering solar panels; light bulb emitting photons; microwaves heating food. In data science, spectral data is used for remote sensing and image analysis.

## Energy Culture

- **Conservation and Conversion**: Total energy is conserved (First Law of Thermodynamics, established 1840s–1850s by Mayer, Joule, Helmholtz). Forms interconvert with efficiencies <100%; "losses" often appear as thermal energy or sound. In data science, this principle underpins loss functions in energy balance models.
- **Storage and Transport**: Examples include batteries (chemical → electrical), pumped hydro (gravitational → kinetic), capacitors/inductors (electrical), and nuclear fuels (nuclear → thermal → mechanical → electrical). Data scientists model these for optimization in supply chain or grid management.

![Energy Conversion Map]({{ '/assets/img/graphics/project/energy/energy_conversion.svg' | relative_url }})

## The International System of Units (SI)

The **International System of Units (SI)**, from the French *Système international d'unités*, is the modern metric system and the world's most widely used measurement standard. It ensures global consistency in science, engineering, industry, and trade. The Bureau International des Poids et Mesures (BIPM) maintains the SI.

### Current Definition (since 2019)
Since 20 May 2019, the SI defines all units by fixing the exact numerical values of seven defining constants of nature. This makes the system universal, stable, and independent of physical artifacts:

- Cesium frequency Δν₁₃₃Cs — 9 192 631 770 Hz
- Speed of light in vacuum c — 299 792 458 m/s
- Planck constant h — 6.626 070 15 × 10⁻³⁴ J s
- Elementary charge e — 1.602 176 634 × 10⁻¹⁹ C
- Boltzmann constant k — 1.380 649 × 10⁻²³ J/K
- Avogadro constant N_A — 6.022 140 76 × 10²³ mol⁻¹
- Luminous efficacy K_cd — 683 lm/W (for monochromatic radiation at 540 THz)

These constants derive the seven base units.

### Seven SI Base Units

- Second (s) — time
- Metre (m) — length
- Kilogram (kg) — mass
- Ampere (A) — electric current
- Kelvin (K) — thermodynamic temperature
- Mole (mol) — amount of substance
- Candela (cd) — luminous intensity

> Use the seven SI base units, and derived units built from them, to enforce consistency and comparability across datasets and models.
{: .prompt-info }

### Brief History of the SI
The metric system originated during the French Revolution. In 1799, platinum standards for the metre and kilogram were established.

The 1875 Metre Convention created the BIPM. In 1960, the 11th CGPM adopted the **International System of Units (SI)** with six base units (mole added 1971). Units were redefined over time using natural constants, culminating in the 2019 revision based entirely on fundamental constants.

### Examples of Derived Units for Energy

- Energy: joule (J = kg·m²·s⁻²)
- Power: watt (W = kg·m²·s⁻³)
- Force: newton (N = kg·m·s⁻²)
- Pressure: pascal (Pa = kg·m⁻¹·s⁻²)

$$
J = \mathrm{kg}\,\mathrm{m}^2\,\mathrm{s}^{-2}
$$

$$
W = \mathrm{kg}\,\mathrm{m}^2\,\mathrm{s}^{-3}
$$

$$
N = \mathrm{kg}\,\mathrm{m}\,\mathrm{s}^{-2}
$$

$$
\mathrm{Pa} = \mathrm{kg}\,\mathrm{m}^{-1}\,\mathrm{s}^{-2}
$$

## Feature Engineering with Dimensions (for Data Scientists Modeling Energy Systems)

Feature engineering is a core step in the data science pipeline, bridging raw data preparation and modeling. A standard feature engineering process typically involves **three key steps**:

1. **Feature Creation**: Generating new features from raw data (e.g., deriving physical quantities, transformations, or interactions).
2. **Feature Transformation**: Applying scaling, normalization, encoding, or other modifications to make features suitable for models.
3. **Feature Selection**: Identifying the most relevant features to reduce dimensionality and improve performance.

In energy-related datasets (e.g., sensor readings, simulation outputs, or experimental logs), incorporating physical dimensions and SI units ensures features are physically meaningful, leading to more interpretable and generalizable models.

- **Unit Harmonization (Preparation for Creation)**: Convert all measurements to SI units using libraries like Pint. This is foundational before creation to avoid inconsistencies.
- **Feature Creation Examples**:
  - Dimensionless groups via Buckingham Π theorem (e.g., Reynolds number for flows).
  - Per-unit quantities (e.g., specific energy J/kg, power density W/m³).
  - Domain-specific: Arrhenius terms, power from $$P = V \cdot I$$, work integrals.
    $$
    k(T) = A \exp\left(-\frac{E_a}{R T}\right)
    $$
- **Feature Transformation Examples**:
  - Log transforms for skewed distributions.
  - Standardization (zero mean, unit variance) or MinMax scaling.
  - Polynomial features or interactions for non-linear relationships.
- **Feature Selection Examples**:
  - Correlation analysis or mutual information to filter redundant features.
  - Recursive Feature Elimination (RFE) or LASSO regularization.
  - Physics-informed selection: Retain features that respect conservation laws.

Advanced: Use physics-informed neural networks (PINNs) to embed constraints during training.

> Convert to SI units first, then build dimensionless and per‑unit features before scaling; this increases robustness across datasets and operating regimes.
{: .prompt-tip }

> Avoid mixing units (e.g., J and kWh) in training; normalize consistently and document unit provenance for every feature.
{: .prompt-warning }

![Feature Engineering Pipeline]({{ '/assets/img/graphics/project/energy/feature_pipeline.svg' | relative_url }})

---

## Practical Example: Battery Discharge Modeling

### Problem Context
Predict remaining useful life (RUL) of a lithium-ion battery from discharge cycle data. Raw sensor data: voltage, current, temperature.

**Raw Data Sample**: Time (min), voltage (V), current (A), temperature (°C) at 5 time points during discharge.

**Problem**: Mixed units, no physical relationships captured.

---

### Step 1: Unit Harmonization (SI Conversion)

Convert all measurements to SI base units:

$$
t[\mathrm{s}] = t[\mathrm{min}] \times 60, \quad T[\mathrm{K}] = T[^\circ\mathrm{C}] + 273.15
$$

**Use Pint or similar library** to enforce consistency across dataset.

---

### Step 2: Feature Creation (Physics-Derived)

#### 2.1 Instantaneous Power
$$
P = V \cdot I \quad [\mathrm{W}] = \mathrm{kg \cdot m^2 \cdot s^{-3}}
$$

Direct application of Ohm's law. Captures energy flow rate.

#### 2.2 Internal Resistance
$$
R = \frac{V}{I} \quad [\Omega] = \mathrm{kg \cdot m^2 \cdot s^{-3} \cdot A^{-2}}
$$

Key degradation indicator—increases as battery ages.

#### 2.3 Cumulative Energy (Numerical Integration)
$$
E(t) = \int_0^t P(\tau) \, d\tau \quad [\mathrm{J}]
$$

Trapezoid rule for discrete data:
$$
E_i = E_{i-1} + \frac{P_i + P_{i-1}}{2} \cdot (t_i - t_{i-1})
$$

```python
# Cumulative energy via trapezoid integration
energy_j = np.zeros(len(time_s))
for i in range(1, len(energy_j)):
    dt = time_s[i] - time_s[i-1]
    p_avg = (power_w[i] + power_w[i-1]) / 2
    energy_j[i] = energy_j[i-1] + p_avg * dt
```

Represents total stress accumulated by the cell.

#### 2.4 Dimensionless Features (State of Charge Proxy)
$$
\text{Voltage Normalized} = \frac{V(t)}{V_{\max}} \quad [\text{dimensionless}]
$$

Scale-invariant indicator. Works across different cell chemistries.

#### 2.5 Per-Unit Features (Specific Power)
$$
P_{\text{specific}} = \frac{P}{m} \quad [\mathrm{W/kg}] = \mathrm{m^2 \cdot s^{-3}}
$$

Normalizes by cell mass. Enables comparison across designs.

#### 2.6 Arrhenius Rate Factor (Thermal Effects)
$$
k(T) = A \exp\left(-\frac{E_a}{RT}\right)
$$

Models temperature-dependent degradation ($E_a \approx 50$ kJ/mol for Li-ion).

```python
# Arrhenius degradation rate
R = 8.314  # J/(mol·K)
E_a = 50000  # J/mol
A = 1e6
arrhenius = A * np.exp(-E_a / (R * temp_k))
```

---

### Step 3: Feature Transformation

#### 3.1 Standardization (Z-score)
$$
z = \frac{x - \mu}{\sigma}
$$

Apply **after** creating physical features. Use `StandardScaler` for power, resistance, specific power.

#### 3.2 Log Transform
$$
\log(1 + E) \quad \text{for skewed energy distributions}
$$

Stabilizes variance for cumulative quantities.

---

### Step 4: Exploratory Data Analysis (EDA)

#### 4.1 Dimensional Consistency Check
Verify all features have correct SI dimensions:
```python
# Check units programmatically
assert power_w.units == 'watt'
assert resistance_ohm.units == 'ohm'
assert energy_j.units == 'joule'
```

#### 4.2 Univariate Analysis

**Distributions**: Plot histograms for each feature
- `power_w`: Check for outliers (sensor errors)
- `resistance_ohm`: Expect gradual increase over cycles
- `temp_k`: Verify physical range (273–373 K typical)

**Summary Statistics**:
$$
\text{Range, Mean, Std, Skewness, Kurtosis}
$$

Flag non-physical values (negative resistance, temp > 400 K).

#### 4.3 Temporal Evolution

Plot time series for physics validation:
- **Power decay**: Should decrease monotonically during discharge
- **Voltage-Current relationship**: Verify $V \propto 1/I$ holds (Ohm's law)
- **Energy accumulation**: Must be monotonically increasing

```python
# Sanity check: energy conservation
assert all(np.diff(energy_j) >= 0), "Energy must increase"
```

#### 4.4 Bivariate Relationships

**Voltage vs. Resistance**: 
$$
R \uparrow \text{ as } V \downarrow \quad \text{(degradation signature)}
$$

**Power vs. Temperature**:
Scatter plot with Arrhenius overlay—expect exponential relationship.

**Correlation Matrix**:
- High $|r|$ between `power_w` and `specific_power` (expected—linear scaling)
- Moderate $|r|$ between `resistance_ohm` and target (degradation link)

#### 4.5 Dimensionless Analysis

Create Buckingham Π groups:
$$
\Pi_1 = \frac{P \cdot t}{E}, \quad \Pi_2 = \frac{R \cdot I}{V}
$$

```python
# Dimensionless feature engineering
pi_1 = (power_w * time_s) / (energy_j + 1e-6)  # avoid division by zero
pi_2 = (resistance_ohm * current_a) / voltage_v
```

**Why**: Plot $\Pi_1$ vs. $\Pi_2$ colored by cycle number—should reveal operating regimes.

#### 4.6 Outlier Detection (Physics-Informed)

Flag measurements violating conservation laws:
- $P > V_{\max} \cdot I_{\max}$ (power exceeds theoretical limit)
- $\Delta E < 0$ (energy decrease impossible)
- $R < 0$ (non-physical resistance)

Use Isolation Forest or LOF on dimensionless features for robustness.

#### 4.7 Feature Importance (Pre-Modeling)

**Mutual Information** with target:
```python
from sklearn.feature_selection import mutual_info_regression
mi_scores = mutual_info_regression(X, y)
```

Expected rankings:
1. `v_normalized` (direct SOC indicator)
2. `resistance_ohm` (degradation)
3. `arrhenius` (thermal stress)

---

### Step 5: Feature Selection (Physics-Informed)

#### Correlation with Target (RUL)
- `v_normalized` (↓ voltage → ↓ RUL)
- `resistance_ohm` (↑ resistance → ↓ RUL)  
- `energy_j` (cumulative stress)

#### Dimensionality Reduction
- Remove redundant features: `power_w` vs. `specific_power` (keep per-unit version)
- Retain dimensionless groups: $\Pi_1$, $\Pi_2$ (scale-invariant)

---

### Final Feature Set for Modeling
```
features = [
    'v_normalized',      # Dimensionless SOC proxy
    'resistance_ohm',    # Internal resistance (degradation)
    'specific_power',    # Per-unit intensity
    'log_energy',        # Transformed cumulative stress
    'arrhenius',         # Temperature-adjusted rate
    'pi_1', 'pi_2'       # Dimensionless groups
]
```

Train with the suitable regressor model. Physical features improve interpretability and generalization.

---

## Key Takeaways

✅ **Start with SI units** → Dimensional consistency  
✅ **Derive physics-based features** → Power, energy, resistance encode real dynamics  
✅ **Perform thorough EDA** → Validate conservation laws before modeling  
✅ **Create dimensionless groups** → Scale-invariant, robust predictions  
✅ **Transform after creation** → Preserve physical meaning before scaling  
✅ **Select with domain knowledge** → Conservation laws guide relevance

---

## Common Pitfalls

❌ **Mixing units**: `energy_kwh + energy_j` → Meaningless  
❌ **Ignoring physics**: Treating $V$ and $I$ independently when $P = VI$  
❌ **Skipping EDA**: Missing outliers that violate conservation laws  
❌ **Premature scaling**: Standardizing before creating ratios  
❌ **Black-box selection**: Dropping `resistance_ohm` despite being known degradation marker
