# Loan Approval Predictor
A machine learning web app built with **Streamlit** that predicts whether a loan application
will be approved based on applicant details such as income and credit score.
---
## Introduction
This project is designed as an experiment to test and compare different machine learning models for the same
prediction task: *loan approval classification*.\
The focus is on **testing and comparing** multiple algorithms (Random Forest, Gradient Boosting, MLP) to understand their
strengths and weaknesses.

Note: This project serves as a practical demo of deploying ML models with Streamlit, while also highlighting the process of experimenting
with different approaches to solve the same problem.
---
## Setup
Clone the repo and install dependencies:
1. Clone repo:
```git clone https://github.com/BlagoTcherecharov/Loan_approval.git```
2. Navigate to project folder
```cd Loan_approval```
3. Install requirements
```pip install -r requirements.txt```
4. Run app
```streamlit run app.py```
---
## Web App
Try the live demo here:
[Loan approval predictor Web app](https://loan-approval-prediction-models.streamlit.app/)
---
## Data
The dataset used for training and testing can be found here:
- Original source(Data + Info): [Kaggle loan prediction dataset](https://www.kaggle.com/datasets/mosaadhendam/loan-prediction-dataset)
- Csv with the data only: [Dataset in github](https://github.com/BlagoTcherecharov/Loan_approval/blob/master/loan_approval_dataset.csv)
---
## Models Tested and Results
All models are implemented using **scikit-learn**:
- Random Forest Classifier
- Gradient Boosting Classifier
- MLP Classifier

The following table summarizes performance on the loan approval dataset:

| Model             | Accuracy | Precision | Recall | F1-score |
|-------------------|----------|-----------|--------|----------|
| Random Forest     | 0.95     | 0.98      | 0.87   | 0.91     |
| Gradient Boosting | 0.998    | 1.00      | 0.99   | 1.00     |
| MLP               | 0.91     | 0.87      | 0.77   | 0.81     |

*Note: Precision, Recall, and F1 are macro averages from `classification_report`.*

---
## Conclusion
Based on the comparison table:
- **Gradient Boosting** delivered the best overall performance.
- **Random Forest** came second, performing reliably but slightly behind boosting.
- **MLP** ranked last, likely due to the relatively small dataset size, which limited the neural network's ability to generalize effectively.

Side note:
- Hyperparameter tuning could improve results for all models (Refer to [Possible/Future Work](#possiblefuture-work)).
- The dataset is relatively small and simulated, so results may not reflect real-world loan approval scenarios.

Overall, ensemble methods (boosting and random forests) proved more effective than the neural network approach for this dataset,
but further tuning and larger, more representative data could change the outcome.
---
## Possible/Future Work
- **Model variety**: Experiment with additional algorithms such as XGBoost, LightGBM, or CatBoost for comparison.
- **Experiment tracking with MLflow**: Log metrics, parameters, and models to compare experiments and manage versions.
---
## License
This project is licensed under the MIT License — see the [LICENSE](https://github.com/BlagoTcherecharov/Loan_approval/blob/master/LICENSE) file for details.