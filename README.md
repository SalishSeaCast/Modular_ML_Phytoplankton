# Modular Machine Learning Framework for Phytoplankton Prediction

[![Tests](https://github.com/SalishSeaCast/Modular_ML_Phytoplankton/actions/workflows/tests.yml/badge.svg)](https://github.com/SalishSeaCast/Modular_ML_Phytoplankton/actions/workflows/tests.yml)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![License](https://img.shields.io/badge/License-MIT-green)

A modular machine learning framework for predicting **diatom production rate** in the Salish Sea. The project demonstrates the transition from a research-oriented notebook workflow to a production-oriented machine learning application through modular software design, reproducible training, model serving with **FastAPI**, automated testing, and continuous integration.

---

# Overview

This repository was developed to transform an exploratory scientific notebook into a reusable and maintainable machine learning framework.

The project demonstrates:

- Modular data processing and feature engineering
- Configuration-driven workflows using YAML
- Reproducible model training
- Model persistence for deployment
- REST API serving using FastAPI
- Automatic API documentation (Swagger & ReDoc)
- Automated testing with Pytest
- Continuous Integration with GitHub Actions

Although the application focuses on phytoplankton prediction, the software architecture is intentionally generic and can be adapted to similar scientific machine learning workflows.

---

# Motivation

Scientific machine learning projects often begin as exploratory notebooks that become increasingly difficult to maintain, reuse, and deploy.

The objective of this project is to bridge the gap between:

- exploratory scientific programming
- reproducible machine learning
- production-oriented software engineering

by applying modern Python development practices while preserving the flexibility required for scientific research.

---

# Features

- Modular project architecture
- YAML-based experiment configuration
- Command Line Interface (CLI)
- Reproducible machine learning pipeline
- Saved deployment-ready model artifacts
- FastAPI inference service
- Dynamic request schema generation
- Automatic OpenAPI documentation
- Logging and error handling
- Unit and integration tests
- GitHub Actions Continuous Integration

---

# Project Structure

```text
Modular_ML_Phytoplankton/

├── .github/
│   └── workflows/
│       └── tests.yml
│
├── configs/
│   └── diat_pr.yaml
│
├── notebooks/
│   ├── diat_pr_hist_old.ipynb
│   └── diat_pr_hist_modular.ipynb
│
├── outputs/
│   ├── logs/
│   └── model/
│       ├── model_diat_pr.joblib
│       └── diat_pr.yaml
│
├── src/
│   ├── analysis.py
│   ├── config.py
│   ├── data.py
│   ├── modeling.py
│   ├── outputs.py
│   ├── processing.py
│   ├── visualization.py
│   │
│   └── api/
│       ├── main.py
│       ├── predictor.py
│       ├── model_loader.py
│       ├── model_info.py
│       └── schemas.py
│
├── tests/
│   ├── test_api.py
│   ├── test_model.py
│   └── test_config.py
│
├── run_analysis.py
├── pyproject.toml
└── README.md
```

---

# Project Architecture

```text
                  YAML Configuration
                         │
                         ▼
                 run_analysis.py
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        ▼                ▼                ▼
   Data Loading     Processing      Feature Engineering
                         │
                         ▼
                  Model Training
                         │
                         ▼
            Saved Pipeline (.joblib)
                         │
         ┌───────────────┴────────────────┐
         ▼                                ▼
   Evaluation                    FastAPI Service
                                          │
                                          ▼
                                 REST Predictions
```

---

# Installation

Clone the repository

```bash
git clone https://github.com/SalishSeaCast/Modular_ML_Phytoplankton.git

cd Modular_ML_Phytoplankton
```

Install the project

```bash
pip install -e .
```

---

# Running the Analysis

Train a model using a configuration file

```bash
python run_analysis.py \
    --config configs/diat_pr.yaml
```

To save the deployment model

```bash
python run_analysis.py \
    --config configs/diat_pr.yaml \
    --save_model
```

The deployment pipeline stores

- trained scikit-learn pipeline (`.joblib`)
- corresponding configuration file

inside

```text
outputs/model/
```

---

# Serving the Model

Start the FastAPI application

```bash
uvicorn src.api.main:app --reload
```

Interactive documentation is automatically available at

Swagger UI

```
http://127.0.0.1:8000/docs
```

ReDoc

```
http://127.0.0.1:8000/redoc
```

The API currently exposes endpoints for

- health check
- model information
- prediction

---

# Testing

Run all tests

```bash
pytest
```

The test suite includes

- model persistence tests
- configuration tests
- API endpoint tests
- prediction endpoint validation

---

# Continuous Integration

GitHub Actions automatically executes the test suite on every push.

The workflow verifies

- package installation
- model loading
- API functionality
- automated tests

ensuring that the repository remains reproducible and functional.

---

# Configuration

Experiments are fully controlled through YAML configuration files located in

```text
configs/
```

Typical configuration options include

- dataset location
- selected environmental drivers
- spatial variables
- temporal variables
- machine learning model
- training parameters
- output options

This allows experiments to be modified without changing the source code.

---

# Future Work

- Docker containerization
- Cloud deployment
- Batch prediction endpoint
- Model versioning
- Performance monitoring
- MLOps integration

---

# License

This project is released under the MIT License.