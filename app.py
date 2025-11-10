import streamlit as st
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

def load_data(path="Crop_Yield_Prediction.csv"):
    df = pd.read_csv(path)
    return df

def train_model(df):
    # Split features/target
    X = df.drop('Yield', axis=1)
    y = df['Yield']

    # One-hot encode the categorical column "Crop"
    X = pd.get_dummies(X, columns=['Crop'], drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Define model and grid
    lr = LinearRegression()
    param_grid = {
        'fit_intercept': [True, False],
        'positive': [True, False]
    }
    grid = GridSearchCV(lr, param_grid, cv=5, scoring='r2')
    grid.fit(X_train, y_train)

    # Evaluate
    best = grid.best_estimator_
    y_pred = best.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    st.write(f"Best parameters: {grid.best_params_}")
    st.write(f"Test R²: {r2:.4f}")
    st.write(f"Test RMSE: {mse**0.5:.2f}")

    # Save model
    joblib.dump(best, "yield_lr_model.joblib")
    return best, X.columns

def main():
    st.title("Crop Yield Prediction App")

    # Load the dataset
    df = load_data()

    # Train the model (you might want to train offline and load instead in production)
    model, feature_columns = train_model(df)

    # Build the prediction form
    st.subheader("Enter input values for prediction")
    nitrogen   = st.number_input("Nitrogen",   min_value=0, step=1, value=90)
    phosphorus = st.number_input("Phosphorus", min_value=0, step=1, value=42)
    potassium  = st.number_input("Potassium",  min_value=0, step=1, value=43)
    temperature= st.number_input("Temperature (°C)", step=0.1, value=21.0)
    humidity   = st.number_input("Humidity (%)",    step=0.1, value=80.0)
    pH_value   = st.number_input("pH Value",         step=0.1, value=6.5)
    rainfall   = st.number_input("Rainfall (mm)",   min_value=0, step=1, value=200)
    crop       = st.selectbox("Crop", ["Maize", "Wheat", "Rice", "Other"])

    if st.button("Predict"):
        # Build sample dataframe with the same feature columns
        sample = pd.DataFrame({
            'Nitrogen': [nitrogen],
            'Phosphorus': [phosphorus],
            'Potassium': [potassium],
            'Temperature': [temperature],
            'Humidity': [humidity],
            'pH_Value': [pH_value],
            'Rainfall': [rainfall],
        })

        # One-hot encode the crop input similarly
        crop_dummies = pd.get_dummies([crop], prefix='Crop', drop_first=True)
        # Add missing dummy columns as zeros
        for col in feature_columns:
            if col.startswith('Crop_') and col not in crop_dummies.columns:
                crop_dummies[col] = 0

        sample = pd.concat([sample, crop_dummies.reset_index(drop=True)], axis=1)
        # Ensure all features present in same order
        sample = sample.reindex(columns=feature_columns, fill_value=0)

        pred = model.predict(sample)[0]
        st.subheader("Prediction result")
        st.write(f"**Predicted Yield:** {pred:.0f}")

if __name__ == "__main__":
    main()
   