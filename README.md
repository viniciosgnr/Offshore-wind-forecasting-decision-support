# Short-Term Offshore Wind Power Forecasting for FPSO Operational Support

## Master's Thesis Project

This repository contains the source code and models developed for the Master's thesis titled: **"[Your Thesis Title Here]"**. The project focuses on developing a high-accuracy, probabilistic forecasting system for offshore wind power and integrating it into a decision support dashboard to enhance the operational efficiency and safety of a Floating Production, Storage and Offloading (FPSO) unit.

---

## Abstract

The reliable integration of offshore wind power into the energy matrix of an FPSO presents a significant operational challenge due to the stochastic nature of wind. This work proposes a comprehensive solution, starting from advanced signal processing techniques and culminating in an AI-powered decision support tool. The core of the solution is a novel hybrid forecasting model, the **CEEMDAN-EWT-TFT Aggregator**, which combines signal decomposition with a Temporal Fusion Transformer to deliver state-of-the-art probabilistic forecasts. These forecasts are then translated into actionable intelligence through a web-based dashboard, which includes risk calibration mechanisms, strategic optimization tools, and LLM-powered agents (Proactive Surveillance and Interactive Copilot) to provide operators with both tactical and strategic guidance. The results demonstrate that this integrated approach can drastically reduce operational costs and mitigate the risk of blackouts, providing a clear blueprint for data-driven decision-making in offshore energy operations.

---

## Features

- **State-of-the-Art Forecasting:** Implements and compares over 15 models, with the proposed **CEEMDAN-EWT-TFT Aggregator** achieving a MAPE of **1.79%** (t+10 min).
- **Probabilistic Predictions:** Generates prediction intervals (e.g., 80% confidence) essential for risk assessment.
- **Interactive Dashboard:** A web-based tool built with Streamlit for real-time simulation, model comparison, and operational analysis.
- **Risk Calibration:** Features a "P10 Safety Factor" to allow operators to tune the system's risk aversion.
- **Strategic Optimizer:** A tool to find the optimal operational threshold and trigger horizon that minimize total costs.
- **LLM-Powered Intelligence:**
    - **Proactive Surveillance Agent:** Autonomously monitors data streams to issue alerts on critical events and trends.
    - **Interactive Operational Copilot:** An on-demand chatbot for tactical and strategic advice, grounded in real-time data via a RAG approach.

---

## Repository Structure

- **`.gitignore`**: Specifies files and folders to be ignored by Git.
- **`README.md`**: This file.
- **`dataset/`**: Contains the raw time series data (tracked with Git LFS).
- **`icons/`**: Icons used in the Streamlit dashboard.
- **`logos/`**: Logos displayed in the dashboard's sidebar.
- **`saved_models/`**: (Ignored by Git) Directory where trained models are saved.
- **`dashboard.py`**: The main source code for the Streamlit application.
- **`myfunctions.py`**: Python script containing all model definitions and helper functions.
- **`requirements.txt`**: Lists all Python dependencies required to run the project.
- **`[Jupyter Notebooks].ipynb`**: Notebooks used for experimentation, analysis, and model development.

---

## Setup and Execution

To run the Decision Support Dashboard on your local machine, please follow these steps.

### 1. Prerequisites

- Python 3.9+
- Git and Git LFS installed.

### 2. Clone the Repository

First, clone the repository. If you are using Git LFS, the large dataset file will be downloaded automatically.

bash
# Clone the repository
git clone https://github.com/seu-usuario/seu-repositorio.git

# Navigate into the project directory
cd seu-repositorio

3. Install Dependencies
It is highly recommended to use a virtual environment.
Bash
# Create a virtual environment
python -m venv venv

# Activate the environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install all required packages
pip install -r requirements.txt
4. Run the Dashboard
Once the dependencies are installed, you can run the Streamlit application.
Bash
streamlit run dashboard.py
The dashboard should automatically open in your web browser.
 
