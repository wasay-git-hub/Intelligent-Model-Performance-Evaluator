# 📈 Rossmann Store Sales Prediction - End-to-End MLOps Pipeline

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange)
![Pytest](https://img.shields.io/badge/Testing-Pytest-green)

## Project Overview
This project implements a modular, reproducible Machine Learning pipeline to predict daily sales for Rossmann drug stores up to 6 weeks in advance, keeping in view multiple factors such as State Holidays, Competitor Stores' Distance, Promotions, etc.

The primary goal of this system is not just to train a model, but to demonstrate **Operational Readiness** by transitioning a raw notebook into a production-grade codebase. It specifically addresses the **Engineering Trade-offs** between model complexity, interpretability, and operational cost.

## Engineering Trade-offs & Model Selection
In this project, we implemented and evaluated three distinct model architectures to analyze the trade-off between **Simplicity** and **Accuracy**.

| Model Architecture | Role | Trade-off Analysis |
| :--- | :--- | :--- |
| **Linear Regression** | *Baseline* | **High Simplicity, Low Accuracy.** Extremely fast training. Highly interpretable coefficients, but failed to capture seasonality, resulting in a high RMSPE (0.59). |
| **Random Forest** | *Challenger* | **Balanced.** Significant improvement over baseline. Parallelizable training. Tuning reduced CV error by **~34%** (0.56 → 0.37), proving the necessity of optimization. |
| **XGBoost** | *Champion* | **High Accuracy, High Complexity.** Achieved the lowest Final Test RMSPE (0.15). Requires careful hyperparameter tuning and dependency management but offers the best predictive performance. |

### Impact of Effective Hyperparameter Tuning
We implemented `RandomizedSearchCV` with **Time Series Split** to optimize the models without data leakage. The results demonstrate that effective tuning is critical for minimizing error:

*   **Random Forest:** Tuning reduced the Validation RMSPE from **0.56 (Baseline)** to **0.37 (Tuned)**, a massive improvement before final testing.
*   **XGBoost:** Tuning further refined the gradient boosting parameters, allowing the model to achieve the state-of-the-art result of **0.15** on the Test set.

## Evaluation Results
The following table summarizes the performance of all experiments. The **RMSPE** (Root Mean Square Percentage Error) is the primary metric used for selection.

| Model | Baseline RMSPE (CV) | Tuned RMSPE (CV) | **Final Test RMSPE** | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Linear Regression** | 0.59 | N/A | **0.46** | Discarded (Underfitting) |
| **Random Forest** | 0.56 | 0.37 | **0.16** | Strong Candidate |
| **XGBoost** | 0.49 | 0.34 | **0.15** | **Production Winner** |

(Note: Standard Linear Regression is a simple parametric model with no hyperparameters to optimize, serving purely as a fixed baseline for performance comparison.)

## Statistical Model Comparison (ANOVA)
To ensure the performance differences between models were not due to random chance, we performed a **One-Way ANOVA (Analysis of Variance)** test on the squared prediction errors of all three models on the Test set.

*   **F-Statistic:** `9518.81`
*   **P-Value:** `0.0000e+00` ($p < 0.05$)

**Conclusion:** The result is **statistically significant**, allowing us to reject the null hypothesis. The analysis confirms that the non-linear architecture of XGBoost provides a fundamental improvement over the baseline, not just a random fluctuation.

| Model | Mean Squared Error (Lower is Better) |
| :--- | :--- |
| **Linear Regression** | 8,098,267.89 |
| **Random Forest** | 1,382,935.45 |
| **XGBoost** | **1,236,260.94** |

## Key Features
*   **Modular Architecture:** Code is separated into logical modules (`preprocessing`, `model`, `evaluation`, `optimization`) within the `src/` package.
*   **Reproducibility:** All hyperparameters, file paths, and split ratios are centralized in `src/params.yaml`.
*   **Automated Tuning:** Implements `RandomizedSearchCV` with **Time Series Split** to prevent data leakage during optimization.
*   **Conditional Logic:** The pipeline only triggers expensive hyperparameter tuning if the baseline model fails to meet the defined performance threshold.
*   **Robust Evaluation:** Optimized for the competition metric **RMSPE** (Root Mean Square Percentage Error).
*   **Deep Error Analysis:** Includes a dedicated notebook (`notebooks/error_analysis.ipynb`) for systematic inspection of failure modes, residual distribution analysis, and     bias detection.
*   **Unit Testing:** `pytest` suite ensures data cleaning logic and feature engineering integrity before training.

## Deep Error Analysis
Conducted a systematic failure analysis to diagnose model weaknesses:
*   **Residual Distribution:** The error residuals follow a normal distribution centered at zero, confirming the model is **statistically unbiased**.
*   **Volatility Impact:** Analysis reveals that **Promotional days** exhibit significantly higher error variance than non-promotional days, indicating the model struggles       with high-volatility sales spikes.
*   **Failure Modes:** The "Worst Predictions" inspection shows the model tends to **under-predict extreme outliers** (e.g., days with >25k sales), suggesting a need for         more aggressive feature engineering for peak events.
  
## Testing Strategy
We use `pytest` to ensure pipeline reliability:
1.  **Schema Checks:** Ensures all required columns exist after merging.
2.  **Logic Checks:** Verifies that closed stores (`Open=0`) are removed and missing values are imputed correctly.
3.  **Leakage Checks:** Ensures the Time-Series Split separates dates chronologically (Train < CV < Test).

## Project Structure
```text
Rossmann-Sales-Pipeline/
├── data/                   # Raw CSV files
│   ├── store.csv
│   └── train.csv
├── models/                 # Serialized models
│   ├── LinearRegression.pkl
│   ├── RandomForest.pkl
│   └── XGBoost.pkl
├── notebooks/              # Experimental analysis
│   └── eda_analysis.ipynb
|   ├── error_analysis.ipynb
├── src/                    # Source code package
│   ├── __init__.py
|   ├── compare_models.py   # Statistical Model Comparison
│   ├── data_loader.py      # Data ingestion logic
│   ├── evaluation.py       # Metrics (RMSPE, MAPE,RMSE, R2)
│   ├── main.py             # Pipeline orchestrator
│   ├── model_serializer.py # Joblib save/load logic
│   ├── model.py            # Model definitions
│   ├── optimization.py     # Hyperparameter tuning logic
│   ├── params.yaml         # Central configuration file
│   ├── preprocessing.py    # Cleaning, Feature Engineering, Splitting
│   └── utils.py            # Config loaders
├── tests/                  # Unit tests
│   ├── __init__.py
│   └── test_pipeline.py    # Tests for preprocessing logic
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

---

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <https://github.com/wasay-git-hub/Intelligent-Model-Performance-Evaluator>
    ```

2.  **Create a virtual environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Configure the Experiment
Open `src/params.yaml` to select the model type or adjust parameters:

```yaml
models:
  type: "XGBoost"  # Options: "Linear Regression", "Random Forest", "XGBoost"
```

### 2. Run Unit Tests
Ensure all logic tests pass.

```bash
pytest tests/
```

### 3. Run the Pipeline
Execute the main script from the **root directory**:

```bash
python src/main.py
```

### 4. Run the ANOVA Test
Run the following command in the terminal:

```bash
python -m src.compare_models
```

The pipeline will Load -> Clean -> Engineer Features -> Split -> Train Baseline -> Evaluate -> (Conditionally Tune) -> Test -> Save Model.
---