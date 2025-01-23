import streamlit as st
import pickle
import pandas as pd
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Load the trained model
model = pickle.load(open("spam_classifier.pkl", "rb"))

# Streamlit application title
st.title("Logistic Regression Prediction App")
st.write("Upload a CSV file for predictions:")

# File uploader for CSV files
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Read the uploaded CSV file
    data = pd.read_csv(uploaded_file)

    # Keep 'PassengerID' and remove other unnecessary columns
    passenger_ids = data['PassengerId']
    columns_to_remove = ['Name', 'Cabin', 'Ticket', 'PassengerId']  # Removed 'PassengerID' from here  # Store PassengerID separately
    data.drop(columns=columns_to_remove, inplace=True, errors='ignore')

    # Fill NaN values with the mean for 'Age' and 'Fare'
    data['Age'].fillna(data['Age'].mean(), inplace=True)
    data['Fare'].fillna(data['Fare'].mean(), inplace=True)

    # Create dummy variables for 'Sex' and 'Embarked'
    if 'Sex' in data.columns and data['Sex'].notnull().any():
        Sex_dummies = pd.get_dummies(data['Sex'], prefix='Sex', drop_first=False)
        data = pd.concat([data, Sex_dummies], axis=1)
        data.drop(columns=['Sex'], inplace=True)

    if 'Embarked' in data.columns:
        embarked_dummies = pd.get_dummies(data['Embarked'], prefix='Embarked', drop_first=False)
        data = pd.concat([data, embarked_dummies], axis=1)
        data.drop(columns=['Embarked'], inplace=True)

    # Ensure the input data has the necessary columns
    required_columns = ["Pclass", "Age", "SibSp", "Parch", "Fare", 
                        "Sex_female", "Sex_male", "Embarked_C", 
                        "Embarked_Q", "Embarked_S"]

    if all(col in data.columns for col in required_columns):
        predictions = model.predict(data[required_columns])
        probabilities = model.predict_proba(data[required_columns])
        
        # Create a DataFrame for the results and include PassengerID
        results = pd.DataFrame({
            "PassengerID": passenger_ids,
            "Predicted Class": ["Survived" if pred == 1 else "Not Survived" for pred in predictions],
            "Probability": [prob[pred] * 100 for prob, pred in zip(probabilities, predictions)]
        })
        
        # Display the results
        st.write("Prediction Results:")
        st.dataframe(results)
    else:
        missing_columns = [col for col in required_columns if col not in data.columns]
        st.error(f"The uploaded file is missing the following columns: {', '.join(missing_columns)}")

if __name__ == '__main__':
    pass