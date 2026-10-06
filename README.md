# Car Price Prediction using Machine Learning

An end-to-end Machine Learning regression pipeline implemented in Python and Jupyter Notebook to predict used and new vehicle prices based on car features (mileage, manufacturer, age, fuel type, etc.).

## 📊 Project Overview
This project processes a comprehensive vehicle advertisement dataset to build predictive regression models. It covers everything from initial exploratory data analysis (EDA) to feature engineering and automated hyperparameter optimization.

The pipeline trains and evaluates three distinct algorithms:
1. **Linear Regression** (Baseline model with interpretation)
2. **K-Nearest Neighbors (kNN) Regressor** (Tuned via GridSearchCV)
3. **Decision Tree Regressor** (Tuned via GridSearchCV)

## 🛠️ Features & Pipeline Architecture
* **Data Cleaning:** Automated handling of missing data using median/most-frequent strategies, outlier clipping via quantiles, and removal of invalid records (e.g., negative mileages).
* **Feature Engineering:** Creates derived features including `car_age` from the registration year, `mileage_per_year`, and simplified categorizations of vehicle conditions.
* **Preprocessing Pipeline:** Implements a strict `ColumnTransformer` layout to cleanly scale numerical fields (`MinMaxScaler`) and encode categorical fields (`OneHotEncoder`) without data leakage.
* **Model Optimization:** Employs a 3-fold cross-validation grid search (`GridSearchCV`) to locate the optimal parameters for non-linear models.

## 💻 Setup and Installation

### Prerequisites
* Python 3.11 or higher
* Standard data science libraries (listed in requirements)

### Installation
1. Clone this repository:
   ```bash
   git clone https://github.com
   cd YOUR_NEW_REPOSITORY_NAME
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 How to Run
You can explore the development phase, data visualizations, and step-by-step evaluations inside the Jupyter Notebook:
```bash
jupyter notebook "main assesment.ipynb"
```
Alternatively, you can run the structured production script directly:
```bash
python main_assessment.py
```

## 📈 Performance Summary
The models are automatically compared on the test set using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R² score metrics. The script automatically isolates the best-performing model and provides a residual distribution analysis.
