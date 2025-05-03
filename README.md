# 🍷 Wine Quality Prediction

> Predicting wine quality using machine learning based on physicochemical tests.

This project demonstrates a practical application of supervised machine learning for binary classification. Using real-world wine quality data, we predict whether a given wine sample is of good or bad quality, based on various chemical attributes.

---

## 🚀 Features

- Clean data preprocessing pipeline
- Exploratory Data Analysis (EDA) with insightful visualizations
- Machine Learning classification using XGBoost and others
- Performance evaluation with metrics and plots
- Easily extensible and reproducible

---

## 📊 Demo

<!-- Replace the links below with your actual images or gifs -->
![EDA Preview](assets/eda_preview.png)
*EDA of alcohol vs wine quality*

![Model Performance](assets/model_performance.png)
*Confusion matrix and classification report*

---

## 🧠 Technologies Used

- Python 3.6+
- pandas, numpy
- seaborn, matplotlib
- scikit-learn
- XGBoost

---

## 🛠️ Installation

Clone the repository:
```bash
git clone https://github.com/AhmedZayed35/001_wine_quality.git
cd 001_wine_quality
```

Install dependencies:


```bash 
pip install -r requirements.txt
```
## 📂 Project Structure
```
001_wine_quality/ 
├── data/
│   └── winequality.csv
├── model.py
├── requirements.txt
└── README.md
```





# Wine Quality Prediction

This repository contains code for predicting wine quality using machine learning algorithms. The dataset used for this project is "winequality.csv", and the goal is to predict whether a wine's quality is above a certain threshold.

## Getting Started

### Prerequisites

To run the code, you need the following prerequisites:

- Python (>=3.6)
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost

You can install the required packages using the requirements.txt file provided in this repository. To install the dependencies, run the following command:

```pip install -r requirements.txt```

### Dataset

The dataset "winequality.csv" should be placed in the "data" directory within the repository.

## Data Preprocessing and Visualization

The code provided in the script "model.py" performs the following data preprocessing and visualization steps:

1. Loading the dataset and checking for missing values.
2. Filling missing values with the mean of each respective column.
3. Visualizing histograms of each feature using matplotlib and seaborn.
4. Creating a bar plot to display the relationship between wine quality and alcohol content.
5. Generating a heatmap to visualize the correlation between features.

## Model Training and Evaluation

The script trains three different machine learning models on the preprocessed data:

1. Logistic Regression
2. XGBoost Classifier
3. Support Vector Classifier (SVC) with RBF kernel

The dataset is split into training and testing sets using the train_test_split function from scikit-learn. The features are then normalized using MinMaxScaler to bring them within the range of [0, 1].

Each model is trained on the training data and evaluated on the testing data using the ROC AUC score. Additionally, the confusion matrix and classification report are generated for the SVC model.
