import joblib
import pandas as pd
import streamlit as st

def interface():
    rf_model = joblib.load("./model_files/random_forest.pkl")
    boosting_model = joblib.load("./model_files/boosting.pkl")
    nn_model = joblib.load("./model_files/neural_network.pkl")

    st.title("ML for loan approval")
    st.write("Enter applicant details below:")

    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    income = st.number_input("Yearly Income", min_value=0, value=50000)
    credit_score = st.slider("Credit Score", 300, 850, 650)
    loan_amount = st.number_input("Loan Amount", min_value=0, value=10000)
    loan_term = st.selectbox("Loan Term (months)", [12, 24, 36, 48, 60])
    employment_status = st.selectbox("Employment Status", ["Employed", "Self-Employed", "Unemployed"])

    X_predict = pd.DataFrame([{
        "Age": age,
        "Income": income,
        "Credit_Score": credit_score,
        "Loan_Amount": loan_amount,
        "Loan_Term": loan_term,
        "Employment_Status": employment_status
    }])

    model_choice = st.sidebar.selectbox(
        "Choose a model:",
        ("Random Forest", "Gradient Boosting", "MLPClassifier")
    )

    if model_choice == "Random Forest":
        clf = rf_model
    elif model_choice == "Gradient Boosting":
        clf = boosting_model
    else:
        clf = nn_model

    if st.button("Submit for Loan Prediction"):
        prediction = clf.predict(X_predict)[0]
        probability = clf.predict_proba(X_predict)[0][prediction]

        # Show prediction result
        if prediction == 1:
            st.success(f"Approved loan with {probability:.1%} confidence")
        else:
            st.error(f"Denied loan with {probability:.1%} confidence")


if __name__ == "__main__":
    interface()