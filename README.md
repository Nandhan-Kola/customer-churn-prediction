📉 Customer Churn Prediction using Machine Learning
Predicting which telecom customers are likely to churn — using real-world data, SMOTE-balanced training, and a Random Forest classifier saved as a deployable model.
________________________________________
🔍 Problem Statement
Customer churn is one of the biggest challenges in the telecom industry. Losing a customer costs far more than retaining one. This project builds a machine learning pipeline that predicts whether a customer will churn based on their demographics, services, and billing information — enabling proactive retention strategies.
________________________________________
📁 Dataset
Source: IBM Watson Telco Customer Churn Dataset (Kaggle)
•	File: WA_Fn-UseC_-Telco-Customer-Churn.csv
•	Rows: 7,043 customers
•	Columns: 21 features + 1 target (Churn)
•	Target: Binary — Yes (churned) / No (retained)
Key features include: tenure, MonthlyCharges, TotalCharges, Contract type, InternetService, PaymentMethod, and more.
________________________________________
🧠 ML Pipeline
1. Data Preprocessing
•	Dropped customerID (not useful for modelling)
•	Fixed TotalCharges column — whitespace entries replaced with 0.0 and cast to float
•	Applied Label Encoding to all categorical features
•	Saved encoders to encoders.pkl for reuse during prediction
2. Handling Class Imbalance
•	Dataset had significant imbalance (~73% No Churn vs ~27% Churn)
•	Applied SMOTE (Synthetic Minority Oversampling Technique) on the training set to balance classes
3. Model Training & Selection
Trained three classifiers with 5-fold cross-validation on the SMOTE-balanced training data:
Model	CV Accuracy
Decision Tree	~78%
Random Forest	~93% ✅
XGBoost	~91%
➡️ Random Forest selected as the best model based on cross-validation accuracy.
4. Model Evaluation
Evaluated on the held-out test set (20% split):
•	Accuracy Score reported on unseen data
•	Confusion Matrix generated
•	Classification Report (Precision, Recall, F1-Score)
5. Predictive System
A prediction pipeline is included that:
•	Loads the saved customer_churn_classifier.pkl model
•	Encodes new customer input using saved encoders.pkl
•	Outputs: Churn / No Churn + probability score
________________________________________
🗂️ Project Structure
customer-churn-prediction/
│
├── WA_Fn-UseC_-Telco-Customer-Churn.csv      # Raw dataset
├── Customer_Churn_Prediction_using_ML.ipynb  # Full notebook
├── customer_churn_classifier.pkl              # Saved Random Forest model
└── README.md
________________________________________
🛠️ Tech Stack
Category	Tools
Language	Python 3
Data Handling	Pandas, NumPy
Visualization	Matplotlib, Seaborn
ML Models	Scikit-learn, XGBoost
Imbalance Handling	imbalanced-learn (SMOTE)
Model Persistence	Pickle
Environment	Google Colab
________________________________________
🚀 How to Run
1.	Clone the repository
2.	git clone https://github.com/YOUR-USERNAME/customer-churn-prediction.git
3.	cd customer-churn-prediction
4.	Install dependencies
5.	pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn
6.	Open the notebook
o	Upload Customer_Churn_Prediction_using_ML.ipynb to Google Colab
o	Place the CSV file in the same directory (or update the file path in the notebook)
o	Run all cells top to bottom
7.	Make a prediction
o	Scroll to Section 7 in the notebook
o	Modify the input_data dictionary with a new customer's details
o	Run the cell to get a Churn / No Churn prediction with probability
________________________________________
💡 Key Takeaways
•	SMOTE significantly improved recall for the minority class (churners)
•	Random Forest outperformed Decision Tree and XGBoost on this dataset
•	Month-to-month contract customers and those with high monthly charges showed higher churn likelihood
•	The trained model is serialized and ready for integration into a web app or API
________________________________________
👤 Author
Kola Nandhan Kumar
B.Tech in Information Technology
📧 kolanandhan@gmail.com
🔗 LinkedIn - Nandhan Kola | GitHub - Nandhan-Kola
________________________________________
⭐ If you found this project useful, consider giving it a star!

