
# Modular ML Phytoplankton

[Build Badge]
[Python Badge]
[License Badge]

A modular and reproducible analysis pipeline for prediction diatom production rate in the Salish Sea, using Python, xarray, and scientific machine learning workflows.

## Overview

This project was developed as part of an effort to transform a research-oriented notebook workflow into a modular, maintainable, and reusable analysis pipeline.

The repository demonstrates:

* Separation of logic and execution
* Configuration-driven workflows using YAML
* Modular data processing and visualization
* Reproducible scientific analyses
* Preparation for future ML, automation, and deployment workflows

## Motivation

The goal of this repository is to bridge the gap between academic scientific coding practices and production-oriented data science workflows.

The project serves as a foundation for future extensions including:

* Functional data analysis
* Machine learning models
* Automated pipelines
* Logging and monitoring
* Cloud deployment
* LLM-assisted analysis workflows

## Project Structure

```text
Modular_ML_Phytoplankton/

├── configs/
│   └── diat_pr.yaml

├── notebooks/
│   ├── diat_pr_hist_old.ipynb
│   └── diat_pr_hist_modular.ipynb

├── src/
│   ├── config.py
│   ├── data.py
│   ├── processing.py
│   ├── analysis.py
│   ├── models.py
│   └── visualization.py

├── run_analysis.py

└── requirements.txt
```

## Architecture Diagram

```text
diat_pr.yaml             
    │
    ▼
run_analysis.py
    │
    ├── data.py
    ├── processing.py
    ├── modeling.py
    ├── evaluation.py
    ├── utils.py
    └── config.py
    │
    ▼
Results


diat_pr.yaml             
    │
    ▼
Notebook
    │
    ├── data.py
    ├── processing.py
    ├── modeling.py
    ├── evaluation.py
    ├── utils.py
    |── visualization.py
    └── config.py
    │
    ▼
Exploration & Visualization
```

## Workflow

1. Load dataset
2. Select predefined spatial regions
3. Train the model
4. Perform analysis
5. Generate visualizations
6. Store and inspect results

## Configuration

Analysis settings are defined through YAML configuration files located in the `configs/` directory.

Examples include:

* Dataset path
* Variable selection
* Region definitions
* Analysis parameters

This allows experiments to be modified without changing the source code.

## Future Work

* Add automated testing
* Introduce structured logging
* Support command-line execution
* Add Docker support
* Implement continuous integration
* Deploy analysis pipelines to cloud environments

```
```
