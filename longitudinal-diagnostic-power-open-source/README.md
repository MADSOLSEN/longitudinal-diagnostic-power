# Wearable Simulation – Longitudinal Screening Framework

This repository contains code supporting the analyses and simulations described in the accompanying manuscript on **longitudinal screening using repeated physiological measurements**, with **narcolepsy type 1 (NT1)** as a motivating case study.

The work demonstrates that for low-prevalence disorders with high night-to-night variability, **single-measurement screening is fundamentally insufficient**, even when specificity is high. By aggregating repeated measurements across nights, false-positive rates can be controlled and clinically meaningful screening performance can be achieved, despite limited per-measurement accuracy.

## Scientific context

Using large polysomnography (PSG) datasets, including repeated recordings, the manuscript evaluates NT1 detection based on physiologically specific REM sleep features (e.g., REM latency and direct REM transitions). While single-night classifiers fail at true population prevalence, simulations show that **multi-night screening can recover diagnostic signal lost to nightly variability**.

To quantify these gains, the study models repeated testing under three probabilistic frameworks representing different degrees of temporal dependence between nights:
- an independent binomial model (optimistic upper bound),
- a first-order Markov model capturing sequential dependence,
- and a Conway–Maxwell–binomial model allowing flexible overdispersion.

Together, these frameworks characterize the trade-offs between monitoring duration, sensitivity, and false-positive burden, and provide a generalizable approach for longitudinal screening beyond NT1, including wearable-based applications.

## Repository scope

This is a **trimmed, open-source version** of an internal research repository. All proprietary data, internal infrastructure, and confidential resources have been removed.

- `src/` — core simulation and analysis code
- `notebooks/` — example and exploratory analyses
- `export/` — generated outputs (ignored by git)

The repository does **not** include raw or derived datasets.

## Usage

Dependencies are listed in `requirements.txt`. Typical usage involves running analysis scripts or notebooks that import functionality from `src/`.

## Citation

If you use this code, please cite the associated manuscript:

> *Diagnostic power emerges from longitudinal physiological measurements*  
> (full citation to be added upon publication)

## License

See `LICENSE`.