---
title: Planning in ML System
description: From Research to Production
date: 2025-03-20
categories:
  - MLOPs
tags:
  - MLOps
  - Machine
  - Learning
  - System
  - Planning
  - Project
  - Management
pin: true
math: true
mermaid: true
---


## Core Agile Principles for ML

Agile allows us to focus on priorities to satisfy customers and deliver efficiently. Time constraints (weeks or days) should be taken seriously, yet ML engineering requires both technical discipline and creative inspiration. We should focus on logs, continuous testing, and technical excellence while ensuring modular code design rather than time-wasting complexity. As the Agile principle states: "Simplicity — the art of maximizing the amount of work not done — is essential."

## Challenges with Traditional Agile in ML

Scrum may not be effective for data science tasks due to unpredictable errors. Model practices should include comparison between base and improved models to avoid Cascading Correction Debt.

## The Kaizen Anti-Pattern

The Kaizen anti-pattern typically unfolds as follows: A problem or improvement potential is identified. Root cause analysis may or may not be completed. A detailed improvement plan with KPIs and activities is created. The initiative starts with enthusiasm, but nothing substantial happens and the organization returns to status quo. Eventually, the improvement plan is forgotten entirely.

This anti-pattern occurs because the true nature of continuous improvement is misunderstood and Lean thinking principles may not be fully embraced.

## Taiichi Ohno's Wisdom on Adaptability

Taiichi Ohno wrote in "Workplace Management": _"If my memory is correct, we were taught that it is a bad thing to give orders in the morning and then change them in the afternoon, but I think that as long as 'the wise mend their ways' and 'the wise man should not hesitate to correct himself,' then we must understand this to mean that we should, in fact, revise the morning's orders in the afternoon."_

This quote emphasizes willingness to change direction quickly based on new information, which is particularly relevant for ML systems requiring immediate adjustments. It highlights the need for very short improvement cycles.

## The Planning-Monitoring Relationship

The critical connection between planning and monitoring requires plans based on monitoring data rather than untested assumptions. Strong feedback loops between implementation and evaluation ensure effective resource allocation based on evidence. Enhanced adaptive capacity through monitoring-informed planning allows systematic identification of successes and failures.

## Robust Deployment and Testing Strategies

Models that perform well in development often fail in production, emphasizing the importance of deployment practices. An adaptive approach implements short improvement cycles with continuous corrections. Regular comparisons continuously benchmark against baseline models. Team empowerment enables quick pivots when model behavior requires changes. Constant monitoring is essential for understanding true model performance, and evidence-based planning means making decisions based on monitoring data, not guesswork.

### Deployment Strategies

Canary deployments release to a small percentage of users first to catch early problems. Blue-green deployments maintain identical environments for instant rollbacks. Rolling updates replace old model instances gradually to reduce risk. A/B testing compares performance between versions for data-driven decisions.

## The Key Insight

ML systems are never "done" - they need continuous monitoring and improvement. Testing evolves after deployment as production reveals unforeseen problems. By connecting monitoring results to planning, we create ML systems that learn and improve over time.