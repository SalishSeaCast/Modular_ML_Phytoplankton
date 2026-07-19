# Modular Machine Learning Framework for Phytoplankton Prediction

[![Tests](https://github.com/SalishSeaCast/Modular_ML_Phytoplankton/actions/workflows/tests.yml/badge.svg)](https://github.com/SalishSeaCast/Modular_ML_Phytoplankton/actions/workflows/tests.yml)
![Python](https://img.shields.io/badge/Python-3.12-blue)
![License](https://img.shields.io/badge/License-MIT-green)

A modular machine learning framework for predicting **diatom production rate** in the Salish Sea. This project demonstrates the transition from an exploratory scientific notebook to a production-oriented machine learning application through modular software design, reproducible training, REST API development, automated testing, containerization, and cloud deployment.

---

# Live Demo

The application is publicly deployed using **Docker** and **Render**.

| Resource | URL |
|----------|-----|
| API | https://modular-ml-phytoplankton.onrender.com |
| Swagger UI | https://modular-ml-phytoplankton.onrender.com/docs |
| ReDoc | https://modular-ml-phytoplankton.onrender.com/redoc |

---

# Overview

The framework provides an end-to-end workflow for scientific machine learning, including:

- Modular data processing and feature engineering
- Configuration-driven experiments (YAML)
- Reproducible model training
- Model persistence
- FastAPI inference service
- Docker containerization
- Automated testing with Pytest
- Continuous Integration with GitHub Actions
- Continuous Deployment with Render

Although developed for phytoplankton prediction, the software architecture is generic and can be adapted to other environmental and scientific ML applications.

---

# Technology Stack

| Category | Technologies |
|----------|--------------|
| Machine Learning | scikit-learn, pandas, xarray |
| Backend | FastAPI, Uvicorn |
| Software Engineering | Docker, GitHub Actions, Pytest, YAML |
| Cloud | Render |

---

# Project Structure

```text
Modular_ML_Phytoplankton/

├── configs/
├── notebooks/
├── outputs/
│   ├── logs/
│   └── model/
├── src/
│   ├── api/
│   ├── analysis.py
│   ├── config.py
│   ├── data.py
│   ├── modeling.py
│   ├── processing.py
│   └── visualization.py
├── tests/
├── run_analysis.py
├── Dockerfile
├── pyproject.toml
└── README.md
```

---

# Architecture

```text
                 YAML Configuration
                        │
                        ▼
                Training Pipeline
                        │
                Saved ML Pipeline
                   (.joblib)
                        │
              FastAPI Prediction API
                        │
                 Docker Container
                        │
        GitHub ──► GitHub Actions
                        │
                        ▼
              Render Cloud Deployment
                        │
                        ▼
                 Public REST API
```

---

# Installation

Clone the repository

```bash
git clone https://github.com/SalishSeaCast/Modular_ML_Phytoplankton.git

cd Modular_ML_Phytoplankton
```

Install locally

```bash
pip install -e .
```

Or build with Docker

```bash
docker build -t modular-ml-phytoplankton .
```

Run the container

```bash
docker run --rm -p 8000:8000 modular-ml-phytoplankton
```

---

# Running the Analysis

Train a model

```bash
python run_analysis.py \
    --config configs/diat_pr.yaml
```

Save a deployment-ready model

```bash
python run_analysis.py \
    --config configs/diat_pr.yaml \
    --save_model
```

The trained model and its configuration are stored in:

```text
outputs/model/
```

---

# Deployment

The application is containerized with Docker and automatically deployed to Render.

The deployment pipeline includes:

- Docker image creation
- Dependency isolation
- Automated GitHub deployment
- Public REST API
- Interactive OpenAPI documentation

---

# Testing

Run the full test suite

```bash
pytest
```

Tests cover:

- configuration loading
- model persistence
- prediction pipeline
- API endpoints

---

# Continuous Integration & Deployment

GitHub Actions automatically validates every push by executing the test suite.

Successful updates to the `main` branch are automatically deployed to the live Render service.

---

# Configuration

Experiments are fully controlled through YAML configuration files located in:

```text
configs/
```

Configuration options include:

- dataset location
- selected predictors
- model parameters
- training options
- output configuration

allowing experiments to be reproduced without modifying source code.

---

# Future Work

- Batch prediction endpoint
- Model versioning
- Performance monitoring
- Automated retraining
- MLOps integration

---

# License

Released under the MIT License.