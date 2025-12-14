---
title: ML System Overview
description: A simple Overview
date: 2025-03-19
categories:
  - MLOPs
tags:
  - MLOps
  - Machine
  - Learning
  - System
  - Model
  - Serving
  - Workflow
  - data-drift
pin: true
math: true
mermaid: true
image:
  path: /assets/img/panels/panel8@4x.png
---
## The Evolving Ecosystem

When we consider ML systems in production environments, we're really discussing complex adaptive systems that require careful orchestration. A machine learning system isn't just about the model itself - it's an intricate ecosystem where data, infrastructure, monitoring, and versioning all interplay.

Think about it like a physical system that continuously evolves. Similar to how physical systems respond to external forces and adapt, machine learning systems deployed in production must adjust to changing data distributions, detect anomalies, and maintain performance over time.


A physical system might include a mass on a 30° inclined plane with gravity and friction, a spring creating natural oscillation with specific characteristics, and an electromagnetic actuator maintaining oscillation when it diminishes. A system like that can be interpretative at first, but it can lead to chaos just by some hidden details that can emerge over time.

![Physical System Analogy](/assets/img/graphics/post_9/gray/m0KtwR1.png){: .dark }
![Physical System Analogy](/assets/img/graphics/post_9/gray/m0KtwR1.png){: .light }
_Figure 1.0: Physical system analogy for ML systems_


## The Critical Role of Time

Time is also a very important component in computer science in general. It lets us see how changes are coming. The most wise people are afraid of time.

What can a wise data scientist do to an ML system, for example in time series?

![Time Series System](/assets/img/graphics/post_9/gray/GTyun46.png){: .dark }
![Time Series System](/assets/img/graphics/post_9/gray/GTyun46.png){: .light }
_Figure 2.0: Time series system with intervention mechanisms_


Consider a time series dataset exhibiting trends, underlying causal variables with foreign key (hidden features) creating oscillatory behavior, and detection and intervention mechanisms that identify when the cyclical pattern deviates from expected behavior. These mechanisms apply immediate intervention (similar to the actuator applying force) and adaptive retraining that adjusts model parameters when drift is detected.

Here we can experiment a lot, and the most outrageous situation can lead us to a very bad amount of time.

## Respecting Time in ML Projects

So you see here what happens to systems. Let's return back and understand what happens if all things respect time.

At first, every project sometimes needs a proof of concept. The time you'll take for the POC is very important. Sometimes we should do it very quickly and maintain it with many details and deadlines like Scrum methodologies that are used in DevOps.

But we know how systems are in ML compared to code. Code is more stable than data, at least if the code isn't brutally forced to change for dependency issues, but most likely not like data or models.

