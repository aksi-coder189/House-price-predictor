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

A baseline regression model used for understanding linear relationships between housing features and house prices.

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

## 📊 Results

| Model                   | R² Score | Performance           |
| ----------------------- | -------- | --------------------- |
| Linear Regression       | 0.9204   | Best Performing Model |
| Random Forest Regressor | 0.8746   | Good Performance      |

The Linear Regression model achieved the highest accuracy and explained approximately **92% of the variation in housing prices**, making it the best-performing model for this dataset.

---

## 📌 Key Findings & Insights

### Feature Impact Analysis

The analysis revealed that **area (total square footage)** is the most influential factor affecting house prices, contributing approximately **84.8%** of the model's explained variance.

Other important contributors include:

- Preferred Area (`prefarea`)
- Number of Bedrooms
- Number of Stories
- Number of Bathrooms

Amenities such as air conditioning, main road access, and basement availability also contribute positively to property value.

### Model Performance

The **Linear Regression model** achieved an **R² Score of 0.9204**, indicating that it explains approximately **92% of the variation in housing prices**, demonstrating strong predictive capability for the given dataset.

### Interesting Observations

- Preferred location (`prefarea`) had a stronger impact on pricing than several physical property attributes.
- Hot water heating showed lower predictive importance than expected.
- Location and property size remain the dominant drivers of house valuation.

### Business Recommendations

Based on the analysis:

- Prioritize properties located in preferred areas.
- Emphasize floor area in marketing and pricing strategies.
- Highlight value-adding amenities such as air conditioning and main road access.
- Use predictive analytics to support pricing decisions and investment planning.

### 📝 Insights & Summary

The analysis showed that **area** is the most influential factor affecting house prices, followed by preferred location (`prefarea`), number of bedrooms, stories, and bathrooms. The **Linear Regression model** achieved an R² score of **0.9204**, meaning it was able to explain about **92% of the variation in house prices**, indicating strong predictive performance. One surprising finding was that **preferred location had a greater impact on pricing than several physical property features**, highlighting the importance of location in real estate. Additionally, hot water heating contributed less to price prediction than expected. Based on these findings, a real estate business should focus on promoting properties in desirable locations and emphasize larger floor areas, as these factors have the strongest influence on property value.

---

## 🏡 Sample Prediction

The project includes a custom house price prediction example where users can modify property features and obtain estimated market prices using the trained model.

---

## 🚀 Future Improvements

* Deploy the model using Streamlit or Flask.
* Build an interactive web-based prediction system.
* Integrate real-time real-estate market data.
* Experiment with advanced ensemble learning models.
* Deploy the model as a cloud-based application.

---

## 📂 Project Structure

```text
House-Price-Prediction/
│
├── analysis.ipynb
├── housepredict.py
├── Housing.csv
├── req.txt
└── README.md
```

---

## 👩‍💻 Author

**Aakriti Singh**
B.Tech Computer Science (Artificial Intelligence)
Machine Learning & Data Analytics Enthusiast

---

⭐ If you found this project useful, consider giving it a star on GitHub.
