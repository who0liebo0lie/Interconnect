# Interconnect
Interconnect is a telecom company exploring predictions of when a clients would leave. Discovering the at risk of leaving clients the company plans to ffer them incentives to stay. 

🔗 Interconnect – Data Analysis & Visualization

This project explores patterns, connections, and trends within a dataset referred to as Interconnect. It includes exploratory data analysis, correlation mapping, and visualization to better understand the structure and insights hidden in the data.

📚 Table of Contents
📌About the Project

🧩 The Challenge

🧪 The Data Journey

⚖️ Target and Data Strategy

🧠 Modeling the Unknown

🏆 The Outcome

🛠 Tools & Stack

🚀 Where It Could Go

💬 Final Thought

Project Structure

Screenshots

Contributing

License

📌 About the Project
# 📡 Predicting Churn Before It Happens: A Machine Learning Story from Interconnect Telecom

Every company wants to grow. But sometimes, **growth starts with retention**.

**Interconnect**, a telecom provider offering landline and internet services, found itself facing a challenge familiar to many in subscription-based industries: **customer churn**. Customers were leaving—but why? And more importantly, could we predict *who* was about to leave and *intervene* before they walked away?

This project uses **machine learning** to answer that question.

---

## 🧩 The Challenge

Interconnect's customers subscribe to various services: cloud backups, streaming TV, antivirus protection, and more. Some pay monthly, others commit to longer contracts. Some use electronic billing, others prefer paper. With all these variables, what behaviors signal an unhappy customer?

We were handed several datasets containing:
- Contract terms  
- Customer demographics  
- Internet & phone services  
- Billing preferences  
- Churn labels  

Each customer had a unique ID, and the data snapshot ended just before **February 1, 2020**.

---

## 🧪 The Data Journey

Before diving into modeling, we needed a **reliable foundation**. I built an `evaluate_file()` function to rapidly assess the shape and health of each input file—checking for:
- Duplicate IDs  
- Null or zero values  
- Data type consistency  
- Random samples for spot-checking  

After standardizing column names and dropping non-numeric outliers (0.002%), all features were converted into boolean or integer format. **Categorical features** like `InternetService`, `PaymentMethod`, and `TechSupport` were mapped to representative numeric codes based on their distribution.

### 🛠 Feature Engineering Highlights
- `customer_tenure`: number of days since joining  
- Total charges accumulated  
- Tech support patterns across service users  
- Multi-line usage in relation to relationship status  

These explorations revealed fascinating signals: for instance, most users *don’t* purchase tech support, but those who do tend to subscribe to streaming services—raising questions about user trust, product quality, or perhaps even demographics.

---

## ⚖️ Target and Data Strategy

Our target was the `churn` column—a binary label.

But there was a problem: **class imbalance**. Many more customers were staying than leaving. To correct this, I wrote a custom **upsampling function** to duplicate churned customers in the training set without inflating performance metrics.

Data was split:
- 60% training  
- 20% validation  
- 20% test  

Features were **standardized with `StandardScaler()`**, ensuring all models operated on uniform input scales.

---

## 🧠 Modeling the Unknown

I built a flexible **machine learning pipeline** to test multiple classifiers:
- Random Forest  
- Decision Tree  
- Logistic Regression  
- Gradient Boosting  
- XGBoost  
- LightGBM  
- CatBoost  

Each model was:
- Tuned via `GridSearchCV`  
- Evaluated using **F1 score**, **ROC-AUC**, and **Precision-Recall Curves**  
- Plotted with training and test metrics for transparency  

---

## 🏆 The Outcome

This project demonstrates how business-critical problems like customer churn can be approached with rigorous machine learning methodology, clear data storytelling, and iterative model evaluation.

Interconnect now has a high-performing, production-ready model that identifies customers at risk of leaving, enabling targeted retention strategies. The Gradient Boosting and CatBoost models emerged as top performers, both achieving an ROC-AUC of 0.85 and precision scores nearing 0.69—demonstrating both discriminative power and reliability.

In analyzing the data, we uncovered several compelling insights:

The average customer lifetime was 32.4 months, contributing on average over $2,036 in revenue.

The longer a customer stayed, the more they spent, confirming that early churn has high revenue cost.

User demographics were equally split by gender, and surprisingly, nearly half of users had multiple lines despite most having no dependents—suggesting possible business usage or bundled plans.

Tech support adoption was low overall (~8,000 users opted out), but streaming customers were more likely to invest in support services than online-only users.

To evaluate model performance, we assessed multiple algorithms using ROC-AUC, F1 Score, Accuracy, and Precision-Recall Curves, while also critically evaluating the strengths and weaknesses of each metric. Given the balanced nature of the dataset, ROC-AUC and Accuracy were deemed the most reliable indicators.

The models evaluated included:

Random Forest, Decision Tree, Gradient Boosting, Logistic Regression, XGBoost, LightGBM, and CatBoost

The final ranking (based on ROC-AUC and Accuracy) crowned Gradient Boosting and CatBoost as tied for first, with Random Forest, LGBM, and Decision Tree closely behind.

This project reflects not just the ability to train performant models, but also the skill of translating model outcomes into actionable business decisions. It's a blend of data engineering, model tuning, and domain insight—the kind of interdisciplinary work that drives real impact.

---

## 🛠 Tools & Stack

- **Languages**: Python  
- **Libraries**: pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost, lightgbm, catboost  
- **Techniques**: feature engineering, upsampling, model evaluation, hyperparameter tuning, scaling

---

## 🚀 Where It Could Go

In a production setting, this model could be deployed as a **REST API**, scoring users daily and triggering automated retention campaigns. It could also be layered with **SHAP or LIME** for interpretability and integrated into Interconnect’s CRM to assist sales and customer service reps.

---

## 💬 Final Thought

This project represents more than a machine learning exercise. It’s a demonstration of **turning business questions into data-driven insights**, using code as the bridge. If you’re a hiring manager looking for engineers who blend technical skill with storytelling and impact orientation—I’d love for you to explore the code behind this.


🚀 Usage
Open the file Interconnect.ipynb in Jupyter Notebook and run all cells in sequence. You’ll walk through:

Data loading and preprocessing

Statistical summaries and visual correlation

Multiple layered plots for exploratory analysis

📁 Project Structure
bash
Copy
Edit
Interconnect.ipynb                    # Main notebook
images_interconnect/                  # Screenshot folder
README.md                             # This file



📸 Screenshots
markdown
Copy
Edit
### 📈 Sample Data Distribution  
![Distribution](images_interconnect/interconnect_image_1.png)

### 📊 Correlation Heatmap  
![Heatmap](images_interconnect/interconnect_image_2.png)

### 📉 Line Trends  
![Line Chart](images_interconnect/interconnect_image_3.png)

### 📦 Feature Histogram  
![Histogram](images_interconnect/interconnect_image_4.png)

### 📌 Boxplot by Category  
![Boxplot](images_interconnect/interconnect_image_5.png)

### 🌀 Pairwise Relationships  
![Pairplot](images_interconnect/interconnect_image_6.png)

### 📍 Scatter Plot Matrix  
![Scatter Matrix](images_interconnect/interconnect_image_7.png)

### 📊 Trend Comparison  
![Trends](images_interconnect/interconnect_image_8.png)

### 📈 Rolling Average Visualization  
![Rolling Avg](images_interconnect/interconnect_image_9.png)

### 📊 Categorical Analysis  
![Categorical](images_interconnect/interconnect_image_10.png)

### 🔁 Residual Diagnostics  
![Residuals](images_interconnect/interconnect_image_11.png)

### 💹 Final Insight Chart  
![Final](images_interconnect/interconnect_image_12.png)

### 🧠 Summary Plot  
![Summary](images_interconnect/interconnect_image_13.png)

🤝 Contributing
If you’d like to extend this project or automate more insight generation, feel free to fork and submit a pull request.

🪪 License
Licensed under the MIT License

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Platform](https://img.shields.io/badge/Platform-JupyterLab%20%7C%20Notebook-lightgrey.svg)
![Status](https://img.shields.io/badge/Status-Exploratory-blueviolet.svg)
![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)

