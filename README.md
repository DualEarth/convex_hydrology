# CONVEX_HYDROLOGY: LSTM Extrapolation Analysis

This repository contains the code and experiments for investigating the behavior of **LSTM internal representations** in hydrological modeling, specifically focusing on **convex hull-based extrapolation analysis**. It explores how novelty and static catchment attributes affect model performance across synthetic and real-world datasets.

---

## Repository Structure and Experiments

The project is structured around two distinct case studies, each contained in its own top-level directory.

### 1. Hawaii_Experiment/ (Synthetic Case Study)

This directory contains code for a **controlled synthetic experiment** likely using National Water Model (NWM) data for a specific region (Hawaii).

* **Goal:** Investigate the fundamental relationship between **extrapolation distance** (distance from the training data convex hull) and **model performance** in a controlled, data-rich setting.
* **Key Files:**
    * `analysis/hull.py`: Code for **Convex Hull** computation and extrapolation distance metrics.
    * `analysis/trend.py`: Scripts for calculating and visualizing the **correlation** between extrapolation and performance.
    * `data_processing/`: Contains utilities (`download.py`, `process.py`) and HPC submission scripts for preparing the NWM forcing and observation data.

### 2. Camels_Experiment/ (Real-World Case Study)

This directory applies the analysis to the widely-used **Catchment Attributes and Meteorology for Large-sample Studies (CAMELS)** dataset.

* **Goal:** **Reinforce** and extend the findings from the synthetic experiment, particularly investigating the role of **static catchment attributes** in restructuring the model's latent space and mitigating extrapolation effects.
* **Key Files:**
    * `camhull.py`: Convex Hull analysis adapted for the CAMELS dataset.
    * `correlation.py`: Scripts to calculate performance-extrapolation correlations across CAMELS.
    * `temporal.py`: Temporal analysis of latent states and extrapolation signals over time.

---

## Infrastructure and Setup

This project leverages the **PyTorch** framework and the **NeuralHydrology** library for training and testing deep learning models (specifically LSTMs) in hydrology.

### Core Components:

* **`training/` & `*Experiment/custom_trainer.py`:** Custom implementations of training and testing routines from the NeuralHydrology library, adapted for specific experimental needs (e.g., saving latent states).
* **`data_processing/`:** Utilities for preparing large-scale hydrological datasets, including:
    * **Data Acquisition:** HPC scripts (`submit_hpc_download.sh`) for efficient data downloading (likely from S3).
    * **Preprocessing:** Scripts (`map_pixel_divide.py`, `process_data.py`) for mapping forcing pixels to catchment boundaries and creating time-series datasets.
* **`environment/`:** Defines the software environment required for the project.
