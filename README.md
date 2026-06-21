# 🏠 House Price Prediction using Machine Learning

## 📌 Project Overview

This project focuses on predicting house prices using Machine Learning techniques based on various property attributes such as area, number of bedrooms, bathrooms, parking spaces, furnishing status, and other amenities.

The objective is to analyze housing data, identify important factors affecting property prices, and build predictive models capable of estimating the market value of residential properties.

---

## 🎯 Objectives

* Perform data preprocessing and feature engineering.
* Conduct Exploratory Data Analysis (EDA) to uncover patterns and relationships.
* Train and evaluate multiple Machine Learning models.
* Compare model performance using standard regression metrics.
* Predict house prices for new property data.

---

## 📊 Dataset

The dataset contains residential housing information, including:

* Area
* Bedrooms
* Bathrooms
* Stories
* Parking
* Main Road Access
* Guest Room Availability
* Basement
* Air Conditioning
* Preferred Area
* Furnishing Status
* House Price

---

## ⚙️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Jupyter Notebook

---

## 🔍 Project Workflow

1. Data Collection & Loading
2. Data Cleaning & Preprocessing
3. Exploratory Data Analysis (EDA)
4. Feature Engineering
5. Model Training
6. Model Evaluation
7. House Price Prediction
8. Result Analysis

---

## 🤖 Machine Learning Models

### Linear Regression

A baseline regression model used for understanding linear relationships between housing features and price.

### Random Forest Regressor

An ensemble learning model used to improve prediction accuracy and capture complex feature interactions.

---

## 📈 Visualizations Included

* Price Distribution Analysis
* Correlation Heatmap
* Actual vs Predicted Comparison
* Feature Importance Analysis
* Model Performance Comparison

---

## 📏 Evaluation Metrics

The models were evaluated using:

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* R² Score

---

## 📌 Key Findings & Insights

### Feature Impact Analysis

The analysis revealed that **area (total square footage)** is the most influential factor affecting house prices, contributing approximately **84.8%** of the model's explained variance. Other important contributors include:

* Preferred Area (`prefarea`)
* Number of Bedrooms
* Number of Stories
* Number of Bathrooms

Amenities such as air conditioning, main road access, and basement availability also contribute positively to property value.

### Model Performance

The **Linear Regression** model achieved an **R² Score of 0.9204**, indicating that it explains approximately **92% of the variation** in housing prices.

The model achieved an average prediction error of approximately **₹0.21 million**, demonstrating strong predictive capability for the given dataset.

### Interesting Observations

One notable finding was that **preferred location (`prefarea`)** had a stronger impact on pricing than several physical property attributes, reinforcing the importance of location in real estate valuation.

Additionally, **hot water heating** showed lower predictive importance than expected compared to other amenities.

### Business Recommendations

Based on the analysis:

* Prioritize properties located in preferred areas.
* Emphasize floor area in marketing and pricing strategies.
* Highlight value-adding amenities such as air conditioning and main road access.
* Use predictive analytics to support pricing decisions and investment planning.
---

## 🏡 Sample Prediction

The project includes a custom house price prediction example where users can modify property features and obtain estimated market prices using the trained model.

---

## 🚀 Future Improvements

* Deploy the model using Streamlit or Flask.
* Integrate real-time real-estate data.
* Experiment with advanced ensemble models.
* Build an interactive web-based prediction system.

---

## 👩‍💻 Author

**Aakriti Singh**

B.Tech Computer Science (AI)

Machine Learning & Data Analytics Enthusiast
