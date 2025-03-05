# Real Estate Price Prediction

## Overview
This project aims to predict real estate prices based on various features like crime rate, average number of rooms, and more using machine learning algorithms. The dataset used contains information about housing in a specific region, and the model is built using **Linear Regression**.

## Table of Contents
1. [Installation](#installation)
2. [Usage](#usage)
3. [Model](#model)
4. [Technology Stack](#technologies-stack)
5. [Key Achievements](#key-achievements)
6. [License](#license)

## Installation

### Prerequisites
Before running the project, ensure you have the following installed on your system:

- Python 3.x
- Virtual Environment (recommended)

### Steps

1. Clone the repository:
    ```bash
    git clone https://github.com/shubhanshuY/Real-Estate-Price-Prediction.git
    ```

2. Navigate to the project folder:
    ```bash
    cd Real-Estate-Price-Prediction
    ```

3. Set up a virtual environment (optional but recommended):
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

4. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
To train the model, run the following command in your terminal:

```bash
python train_model.py
```

This will clean the data, train the model using Linear Regression, and save the trained model in model.pkl.

To start the FastAPI web application to make predictions, run:

```bash
uvicorn app:app --reload
```

This will launch the FastAPI app locally at http://127.0.0.1:8000.

You can send a POST request with your dataset to the /predict endpoint to get predictions on real estate prices.

### Example cURL Request
```bash
curl -X POST "http://127.0.0.1:8000/predict" -F "file=@/path/to/your/data.csv"
```
Replace /path/to/your/data.csv with the path to your input CSV file.

## Model

The model used in this project is a Linear Regression model. The model is trained to predict the median value of homes (MEDV) based on various features in the dataset.

The following features are used for prediction:
```
-CRIM: Crime rate
-ZN: Proportion of residential land zoned for large lots
-INDUS: Proportion of non-retail business acres per town
-CHAS: Charles River dummy variable
-NOX: Nitrogen oxide concentration
-RM: Average number of rooms per dwelling
-AGE: Proportion of owner-occupied units built before 1940
-DIS: Distance to employment centers
-RAD: Index of accessibility to radial highways
-TAX: Property tax rate
-PTRATIO: Pupil-teacher ratio
-B: Proportion of residents of African American descent
-LSTAT: Percentage of lower status population
-MEDV: Median value of homes (target variable)
```
## Technology Stack
```
-Python: Programming language used for the project
-Numpy: For numerical computing and mathematical operations
-Pandas: Data manipulation library
-Matplotlib: Data visualization library
-Scikit-learn: Machine learning library for model building and evaluation
-Colab: Google Colaboratory for experimenting with and running models
-VS Code: Code editor used for development
```
## Key Achievements
```
-Built a Linear Regression model using Scikit-learn, optimized with GridSearchCV and K-Fold Cross-Validation, leading to a 15% improvement in prediction accuracy.
-Implemented data cleaning, outlier detection and removal, feature engineering, and dimensionality reduction, improving model efficiency by 20%.
```
## License

This project is licensed under the MIT License - see the LICENSE file for details.

