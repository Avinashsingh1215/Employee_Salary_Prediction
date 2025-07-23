# 💼 Employee Salary Prediction using Machine Learning

This project uses various machine learning models to predict whether an individual's income exceeds \$50K/year based on census data from the UCI Adult dataset.

## 📌 Problem Statement

The goal is to predict whether a person earns more than \$50K/year based on features such as education, occupation, marital status, and other demographic attributes. This binary classification task can help in automating decision-making processes in HR analytics and finance domains.

---

## 🛠️ System Development Approach

1. **Data Preprocessing**:
   - Loaded the adult.csv Dataset.
   - Handled missing values and performed label encoding.
   - Applied one-hot encoding for categorical variables.
   - Scaled numerical features using `StandardScaler`.

2. **Model Building**:
   - Applied multiple classification models using pipelines.
   - Models compared: Logistic Regression, Random Forest, K-Nearest Neighbors (KNN), Support Vector Machine (SVM), and Gradient Boosting.

3. **Evaluation Metrics**:
   - Accuracy Score
   - Classification Report (Precision, Recall, F1-Score)
   - ROC-AUC Curve
   - Confusion Matrix
   - Feature Importance (XGBoost-based analysis)

---

## ⚙️ Algorithm & Deployment

| Model               | Accuracy | AUC Score |
|--------------------|----------|-----------|
| Logistic Regression|  0.81    | 0.86      |
| Random Forest      |  0.85    | 0.91      |
| KNN                |  0.82    | 0.85      |
| SVM                |  0.83    | 0.89      |
| Gradient Boosting  |  0.86    | 0.92      |

- ROC Curves were plotted to visualize model performance.
- Pipelines were used to combine scaling and modeling efficiently.

---

## 📊 Result

- **Best Performing Model:** Gradient Boosting (AUC: 0.92)
- **Top Features Identified:** Education, Capital Gain, Hours per Week
- Models were evaluated using AUC-ROC and Confusion Matrix for better interpretation.

---

## ✅ Conclusion

This project demonstrates that advanced ensemble models like **Random Forest** and **Gradient Boosting** outperform basic classifiers in predicting income level. Proper data preprocessing and model tuning significantly improve model accuracy and generalizability.

---

## 🚀 Future Scope

- Integrate hyperparameter tuning using GridSearchCV or RandomizedSearchCV.
- Deploy the model via a Flask API for real-time prediction.
- Extend the project to a web dashboard using Streamlit.
- Perform SHAP analysis for explainability of predictions.

---

## 📚 References

- I am Using the dataset which is provided during My Internship at Edunet Foundation the dataset is adult.csv 
- Scikit-learn Documentation: https://scikit-learn.org/
- Matplotlib & Seaborn for visualization

---

## 🧠 Author

**Avinash Singh**

Thanks to **Edunet Foundation** for this internship opportunity and support in practical learning through this project.

---

## 🔗 Connect with Me

- [GitHub](https://github.com/Avinashsingh1215)

