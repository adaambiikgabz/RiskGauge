import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Define the mapping dictionary
mapping_dict = {
    'Home':{
        'OWN': 0,
        'RENT': 1,
        'MORTGAGE': 2,
        'OTHER': 3
    },
    'Intent':{
        'PERSONAL': 0,
        'EDUCATION': 1,
        'MEDICAL': 2,
        'VENTURE': 3,
        'HOMEIMPROVEMENT': 4,
        'DEBTCONSOLIDATION': 5
    }
}

# Function to map categorical features
def map_categorical_features(data, mappings):
    for column, mapping in mappings.items():
        if column in data.columns:
            try:
                data[column] = data[column].replace(mapping)
            except Exception as e:
                st.error(f"An error occurred when mapping column '{column}': {e}")
        else:
            st.warning(f"The column '{column}' is not found in the DataFrame")
    return data

# Load the saved best model
@st.cache_resource
def load_model():
    model = joblib.load('best_model_GradientBoostingRegressor.pkl')
    return model

model = load_model()

# Define the feature columns (excluding 'Default' and 'Id')
feature_columns = ['Age', 'Income', 'Home', 'Emp_length', 'Intent', 
                   'Amount', 'Rate', 'Status', 'Percent_income', 'Cred_length']

# Sidebar for navigation
st.sidebar.title("Loan Default Prediction")
st.sidebar.write("Choose an option to input data:")

options = ["Upload CSV", "Manual Input"]
choice = st.sidebar.radio("Select Input Method", options)

def main():
    st.title("Loan Default Prediction App")
    st.write("""
    This application predicts the likelihood of a loan default based on the input features.
    You can either upload a CSV file containing the data or manually input the feature values.
    """)

    if choice == "Upload CSV":
        st.header("Upload your CSV file")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
                st.success("File uploaded successfully!")
                st.write("### Uploaded Data:")
                st.dataframe(data.head())

                # Preprocess the data
                data_processed = map_categorical_features(data.copy(), mapping_dict)

                # Ensure all feature columns are present
                missing_cols = set(feature_columns) - set(data_processed.columns)
                if missing_cols:
                    st.error(f"The following required columns are missing: {missing_cols}")
                    return

                X = data_processed[feature_columns]

                # Make predictions
                predictions = model.predict(X)
                data['Default_Prediction'] = predictions

                st.write("### Prediction Results:")
                st.dataframe(data[['Id', 'Default_Prediction']].head())

                # Option to download the results
                csv = data.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Predictions as CSV",
                    data=csv,
                    file_name='predictions.csv',
                    mime='text/csv',
                )

            except Exception as e:
                st.error(f"Error processing the file: {e}")

    elif choice == "Manual Input":
        st.header("Input Feature Values")
        with st.form(key='input_form'):
            Age = st.number_input("Age", min_value=18, max_value=100, value=30)
            Income = st.number_input("Income", min_value=0, value=50000)
            Home = st.selectbox("Home Status", options=['OWN', 'RENT', 'MORTGAGE', 'OTHER'])
            Emp_length = st.number_input("Employment Length (years)", min_value=0, max_value=50, value=5)
            Intent = st.selectbox("Loan Intent", options=['PERSONAL', 'EDUCATION', 'MEDICAL', 
                                                          'VENTURE', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION'])
            Amount = st.number_input("Loan Amount", min_value=0, value=10000)
            Rate = st.number_input("Interest Rate", min_value=0.0, max_value=100.0, value=5.5)
            Status = st.number_input("Credit Score Status", min_value=300, max_value=850, value=700)
            Percent_income = st.number_input("Percentage of Income", min_value=0.0, max_value=100.0, value=20.0)
            Cred_length = st.number_input("Credit History Length (months)", min_value=0, value=24)

            submit_button = st.form_submit_button(label='Predict')

        if submit_button:
            try:
                input_data = {
                    'Age': Age,
                    'Income': Income,
                    'Home': Home,
                    'Emp_length': Emp_length,
                    'Intent': Intent,
                    'Amount': Amount,
                    'Rate': Rate,
                    'Status': Status,
                    'Percent_income': Percent_income,
                    'Cred_length': Cred_length
                }

                input_df = pd.DataFrame([input_data])

                # Map categorical features
                input_df_mapped = map_categorical_features(input_df, mapping_dict)

                # Make prediction
                prediction = model.predict(input_df_mapped)
                prediction_proba = model.predict_proba(input_df_mapped) if hasattr(model, "predict_proba") else None

                # Display the result
                if prediction_proba is not None:
                    st.write(f"### Predicted Default Status: **{prediction[0]}**")
                    st.write("### Prediction Probabilities:")
                    prob_df = pd.DataFrame(prediction_proba, columns=['Prob_Not_Default', 'Prob_Default'])
                    st.dataframe(prob_df)
                else:
                    st.write(f"### Predicted Default Status: **{prediction[0]}**")

            except Exception as e:
                st.error(f"Error during prediction: {e}")

if __name__ == "__main__":
    main()
