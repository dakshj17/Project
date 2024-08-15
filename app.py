# app.py
import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV

# Load your dataset
df = pd.read_csv('data_v7.csv')

# Clean up column names
df.columns = df.columns.str.strip().str.replace(' ', '_')

# Prepare features (X) and target (Y)
X = df.drop(columns=["Talk_n_TXT", "Total_Talk_Time(in_min)", "SMS_sent",
                     "Prepaid/Postpaid", "Company", "Tier", "Plan",
                     "Included_Data(in_GB)"])
Y = pd.get_dummies(df["Plan"])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Train the XGBoost model
model = xgb.XGBClassifier(objective='binary:logistic', random_state=42)
model.fit(X_train, y_train)

# Define parameter grid for GridSearchCV
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.1, 0.01, 0.001],
    'n_estimators': [100, 200, 300]
}

# Perform Grid Search CV to find the best model
grid_search = GridSearchCV(estimator=model, param_grid=param_grid,
                           scoring='accuracy', cv=5, verbose=0)  # Set verbose=0 to suppress fitting details
grid_search.fit(X_train, y_train)

# Get the best model from Grid Search CV
model_new = grid_search.best_estimator_

# Function to handle user input and predict plan
def predict_plan(Cost, Data, Strength):
    Z = pd.DataFrame({"Cost(AUD)": [Cost],
                      "Total_Consumption_Data(in_GB)": [Data],
                      "CoverageStrength": [Strength]})

    # Predict the plan based on user input
    predicted_class = model_new.predict(Z)
    predicted_class_index = np.argmax(predicted_class)  # Get the index of the predicted class

    # Get the name of the predicted plan from Y.columns
    predicted_plan = Y.columns[predicted_class_index]

    # Retrieve additional details of the recommended plan
    row_index = df['Plan'] == predicted_plan  # Find the index of the predicted class
    details = {
        'Plan': predicted_plan,
        'Prepaid/Postpaid': df.loc[row_index, 'Prepaid/Postpaid'].values[0],
        'Company': df.loc[row_index, 'Company'].values[0],
        'Tier': df.loc[row_index, 'Tier'].values[0],
        'Included Data(in GB)': df.loc[row_index, 'Included_Data(in_GB)'].values[0],
        'Cost(AUD)': df.loc[row_index, 'Cost(AUD)'].values[0]
    }

    return details

# Streamlit UI
st.title('Plan Recommendation System')

cost = st.slider('Cost of plan (AUD):', 0.0, df['Cost(AUD)'].max(), step=1.0)
data_usage = st.slider('Estimated data usage (GB):', 0.0, df['Total_Consumption_Data(in_GB)'].max(), step=1.0)
coverage_strength = st.slider('Coverage strength:', 0.0, 1.0, step=0.01)

if st.button('Predict'):
    result = predict_plan(cost, data_usage, coverage_strength)

    st.write("**According to your preferences, your recommended plan is:**")
    st.write("**Plan:**", result['Plan'])
    st.write("**Prepaid/Postpaid:**", result['Prepaid/Postpaid'])
    st.write("**Company:**", result['Company'])
    st.write("**Tier:**", result['Tier'])
    st.write("**Included Data(in GB):**", result['Included Data(in GB)'])
    st.write("**Cost(AUD):**", result['Cost(AUD)'])
