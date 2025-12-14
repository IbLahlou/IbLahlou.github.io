---
title: Forms of Energy and the International System of Units (SI)
description: An overview of the main forms of energy, their interconnections, measurement using SI units, and practical considerations for modeling and feature engineering in data science applications
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
pin: true
math: true
mermaid: true
image:
  path: /assets/img/panels/panelx@4x.png
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
- 
$$
J = \mathrm{kg}\,\mathrm{m}^2\,\mathrm{s}^{-2}
$$
-
$$
W = \mathrm{kg}\,\mathrm{m}^2\,\mathrm{s}^{-3}
$$
-
$$
N = \mathrm{kg}\,\mathrm{m}\,\mathrm{s}^{-2}
$$
-
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
