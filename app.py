import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.impute import SimpleImputer

# Title of the app
st.title("Simple Regression Modeling App")

# Introduction and Theory
with st.expander("Model Theory and Explanations"):
    st.write("""
    ### Model Options:
    1. **Linear Regression**: This model assumes a linear relationship between the target variable and predictor variables. The formula for a linear regression model is:
       
       $$ y = b_0 + b_1x_1 + b_2x_2 + \dots + b_nx_n $$
       
       where:
       - \( y \) is the target variable
       - \( x_1, x_2, \dots, x_n \) are the predictor variables
       - \( b_0 \) is the intercept
       - \( b_1, b_2, \dots, b_n \) are the coefficients for each predictor variable
    
    2. **Random Forest Regression**: A nonlinear model that uses multiple decision trees to make predictions. It captures complex relationships between variables but doesn’t have a fixed formula like linear regression.
    """)

    st.write("""
    ### Model Fit Metrics:
    - **R-squared** (\( R^2 \)): Measures how well the model explains the variance in the target variable, ranging from 0 to 1. Higher values indicate a better fit, with values above 0.7 generally considered good.
    - **Mean Absolute Error (MAE)**: The average of absolute errors between predicted and actual values. Lower MAE indicates a more accurate model, with smaller errors on average.
    """)

# Step 1: File Upload
st.sidebar.header("Upload your Excel file")
uploaded_file = st.sidebar.file_uploader("Choose an Excel file", type="xlsx")

if uploaded_file:
    # Load the data
    data = pd.read_excel(uploaded_file)
    st.write("Dataset Preview:", data)
    
    # Handle Missing Values
    st.sidebar.header("Missing Value Handling")
    missing_value_method = st.sidebar.selectbox("Choose a method to handle missing values", 
                                                ["Drop rows with NaN", "Fill with mean", "Fill with median", "Fill with zero"])

    # Select only numeric columns for further analysis
    numeric_data = data.select_dtypes(include=np.number)
    
    if missing_value_method == "Drop rows with NaN":
        numeric_data = numeric_data.dropna()
    else:
        imputer_strategy = 'mean' if missing_value_method == "Fill with mean" else ('median' if missing_value_method == "Fill with median" else 'constant')
        fill_value = 0 if missing_value_method == "Fill with zero" else None
        imputer = SimpleImputer(strategy=imputer_strategy, fill_value=fill_value)
        numeric_data = pd.DataFrame(imputer.fit_transform(numeric_data), columns=numeric_data.columns)
    
    # Display Correlation Matrix for Numeric Data Only
    if st.checkbox("Show Correlation Matrix"):
        correlation_matrix = numeric_data.corr()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    
    # Step 2: Select Target Column
    numeric_columns = numeric_data.columns.tolist()
    target_column = st.sidebar.selectbox("Select the Target Column", numeric_columns)
    
    # Step 3: Select Predictor Columns
    predictor_columns = st.sidebar.multiselect("Select Predictor Variables", [col for col in numeric_columns if col != target_column])

    if predictor_columns:
        # Step 4: Choose Regression Type
        regressor_type = st.sidebar.selectbox("Choose Regression Model", ["Linear Regression", "Nonlinear Regression (Random Forest)"])

        st.sidebar.write("")
        if st.sidebar.checkbox("Start Regression!"):

            # Split the data
            X = numeric_data[predictor_columns]
            y = numeric_data[target_column]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Initialize the model
            if regressor_type == "Linear Regression":
                model = LinearRegression()
            else:
                model = RandomForestRegressor(random_state=42)

            # Train the model
            model.fit(X_train, y_train)

            # Step 5: Model Evaluation Metrics
            st.divider()
            st.subheader("Model Fit Metrics")
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            st.write(f"**R-squared:** {r2:.2f} (Higher values indicate a better fit, >0.7 is generally good)")
            st.write(f"**Mean Absolute Error:** {mae:.2f} (Lower values indicate better accuracy)")

            # Predicted vs. Actual Plot
            st.divider()
            st.subheader("Predicted vs Actual Values")
            fig, ax = plt.subplots()
            ax.scatter(y_test, y_pred, alpha=0.7)
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            st.pyplot(fig)

            # Step 6: Calculate Variable Contribution
            st.divider()
            st.subheader("Variable Contribution to Target")

            if regressor_type == "Linear Regression":
                contributions = pd.DataFrame({
                    "Variable": predictor_columns,
                    "Contribution": model.coef_
                }).sort_values(by="Contribution", ascending=False)
            else:
                contributions = pd.DataFrame({
                    "Variable": predictor_columns,
                    "Contribution": model.feature_importances_
                }).sort_values(by="Contribution", ascending=False)

            st.write(contributions)

            # Feature Importance Plot
            st.subheader("Feature Importance")
            fig, ax = plt.subplots()
            sns.barplot(x="Contribution", y="Variable", data=contributions, ax=ax)
            st.pyplot(fig)

            # Step 7: Show Individual Predictor Curves
            st.divider()
            st.subheader("Predictor vs Target Variable Relationships")
            for col in predictor_columns:
                fig, ax = plt.subplots()
                sns.scatterplot(x=numeric_data[col], y=numeric_data[target_column], ax=ax)
                ax.set_title(f"{col} vs {target_column}")
                ax.set_xlabel(col)
                ax.set_ylabel(target_column)
                st.pyplot(fig)

            # Step 8: Make Predictions based on User Input
            st.divider()
            st.subheader("Predict Target Variable")

            input_values = []
            for col in predictor_columns:
                input_value = st.number_input(f"Input value for {col}", value=float(X[col].mean()))
                input_values.append(input_value)

            prediction_input = np.array(input_values).reshape(1, -1)
            prediction = model.predict(prediction_input)[0]

            st.write("")

            st.metric(label="Predicted " + target_column, value=prediction.round(3), delta=None)

            #st.write(f"Predicted {target_column}: {prediction}")
