# Telco Customer Churn Prediction

## 1. Business Problem

Customer churn is a major challenge in the telecommunications industry. When customers cancel their subscriptions, the company loses recurring revenue.

Acquiring new customers is often more expensive than retaining existing ones. Therefore, identifying customers who are likely to churn is critical for implementing effective retention strategies.

The objective of this project is to build a machine learning model that predicts whether a customer is likely to churn. With this information, the company can take proactive actions such as:

- Targeted marketing campaign for specific segement of the customer
- Personalized offers or discounts campaigns
- Improved service.

Early identification of churn risk can help reduce revenue loss and improve customer lifetime value.

---

## 2. Problem Formulation

The business problem is translated into a machine learning task as follows.

**Task Type**

Binary Classification

**Objective**

Predict whether a customer will churn.

**Target Variable**
Churn
1 = customer leaves
0 = customer stays


The model learns patterns from historical customer data to estimate the probability of churn.

---

## 3. Dataset Description

Dataset used: **Telco Customer Churn Dataset**

**Dataset Size**
7043 customers


**Feature Categories**

### Demographic Information

- Gender
- SeniorCitizen
- Partner
- Dependents

### Service Information

- PhoneService
- InternetService
- OnlineSecurity
- StreamingTV
- StreamingMovies

### Account Information

- Contract
- Tenure
- MonthlyCharges
- TotalCharges
- PaymentMethod

**Target Variable**
Churn


---

## 4. Exploratory Data Analysis (EDA)

EDA is performed to understand the dataset structure and identify patterns related to churn.

Goals of EDA:

- Understand feature distributions
- Detect missing values
- Identify outliers
- Explore relationships between features and churn

---

### Dataset Overview

Initial inspection includes:

- Dataset shape 
- Data types
- Missing values
- Summary statistics
- related features
