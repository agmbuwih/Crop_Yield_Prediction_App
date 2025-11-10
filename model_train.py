import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import seaborn as sns

# For modelling
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

# Regressors
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

import joblib

def main():
    # --- Load dataset ---
    df = pd.read_csv('/content/Crop_Yield_Prediction.csv')  # adjust path as necessary

    # --- Basic checks ---
    print(df.head())
    print(df.info())
    print("Duplicates:", df.duplicated().sum())
    print("Missing values:\n", df.isnull().sum())
    print("Unique Crops:", df['Crop'].unique())
    print(df.describe())

    # --- Feature & target separation ---
    X = df.drop('Yield', axis=1)
    y = df['Yield']

    numeric_features = ['Nitrogen','Phosphorus','Potassium','Temperature','Humidity','pH_Value','Rainfall']
    categorical_features = ['Crop']

# Define how to process each type
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Combine them into one preprocessor
preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
 )
return preprocessor

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- Baseline model ---
# Baseline Model

baseline_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', DummyRegressor(strategy='mean'))
])

baseline_pipeline.fit(X_train, y_train)

y_pred_baseline = baseline_pipeline.predict(X_test)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae_baseline = mean_absolute_error(y_test, y_pred_baseline)
rmse_baseline = np.sqrt(mean_squared_error(y_test, y_pred_baseline))
r2_baseline = r2_score(y_test, y_pred_baseline)
mape_baseline = mean_absolute_percentage_error(y_test, y_pred_baseline)

print("Baseline Model Performance (DummyRegressor):")
print(f"MAE: {mae_baseline:.2f}")
print(f"RMSE: {rmse_baseline:.2f}")
print(f"R²: {r2_baseline:.2f}")
print(f"MAPE: {mape_baseline*100:.2f}")

 # Build pipeline with LinearRegression
lr_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])
lr_param_grid = {
    'regressor__fit_intercept': [True, False],
    'regressor__positive': [True, False]  # works only for non-negative targets
}
lr_grid = GridSearchCV(estimator=lr_pipeline, param_grid=lr_param_grid, cv=5)

# Train
lr_grid.fit(X_train, y_train)

best_lr_pipeline = lr_grid.best_estimator_

# Predict
y_pred_lr = best_lr_pipeline.predict(X_test)

# Evaluate
mae_lr = mean_absolute_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
r2_lr = r2_score(y_test, y_pred_lr)
mape_lr = mean_absolute_percentage_error(y_test, y_pred_lr)

print("Linear Regression Performance:")
print(f"MAE: {mae_lr:.2f}")
print(f"RMSE: {rmse_lr:.2f}")
print(f"R²: {r2_lr:.2f}")
print(f"MAPE: {mape_lr*100:.2f}")

    # Random Forest and Gradient Boosting with GridSearch 
    # Tandom Forest
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

# Hyperparameter grid
rf_param_grid = {
    'regressor__n_estimators': [100, 200],
    'regressor__max_depth': [None, 10, 20],
    'regressor__min_samples_split': [2, 5]
}

rf_grid = GridSearchCV(rf_pipeline, rf_param_grid, cv=5, scoring='r2', n_jobs=-1)
rf_grid.fit(X_train, y_train)
print("Best RF params:", rf_grid.best_params_)

y_pred_rf = rf_grid.predict(X_test)
print("Random Forest Performance:")
print(f"MAE: {mean_absolute_error(y_test, y_pred_rf):.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_rf)):.2f}")
print(f"R²: {r2_score(y_test, y_pred_rf):.2f}")
print(f"MAPE: {mean_absolute_percentage_error(y_test, y_pred_rf)*100:.2f}")

# Gradient Boosting
gb_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', GradientBoostingRegressor(random_state=42))
])

gb_param_grid = {
    'regressor__n_estimators': [100, 200],
    'regressor__learning_rate': [0.1, 0.05],
    'regressor__max_depth': [3, 5]
}

gb_grid = GridSearchCV(gb_pipeline, gb_param_grid, cv=5, scoring='r2', n_jobs=-1)
gb_grid.fit(X_train, y_train)
print("Best GB params:", gb_grid.best_params_)

y_pred_gb = gb_grid.predict(X_test)
print("Gradient Boosting Performance:")
print(f"MAE: {mean_absolute_error(y_test, y_pred_gb):.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_gb)):.2f}")
print(f"R²: {r2_score(y_test, y_pred_gb):.2f}")
print(f"MAPE: {mean_absolute_percentage_error(y_test, y_pred_gb)*100:.2f}")

# Collect performance metrics
results = {
    'Model': ['Baseline (DummyRegressor)', 'Linear Regression', 'Random Forest', 'Gradient Boosting'],
    'MAE': [mae_baseline, mae_lr, mean_absolute_error(y_test, y_pred_rf), mean_absolute_error(y_test, y_pred_gb)],
    'RMSE': [rmse_baseline, rmse_lr, np.sqrt(mean_squared_error(y_test, y_pred_rf)), np.sqrt(mean_squared_error(y_test, y_pred_gb))],
    'R²': [r2_baseline, r2_lr, r2_score(y_test, y_pred_rf), r2_score(y_test, y_pred_gb)],
    'MAPE (%)': [mape_baseline*100, mape_lr*100, mean_absolute_percentage_error(y_test, y_pred_rf)*100, mean_absolute_percentage_error(y_test, y_pred_gb)*100]
}

results_df = pd.DataFrame(results)

# Sort by R² descending (best performance at top)
results_df = results_df.sort_values(by='R²', ascending=False).reset_index(drop=True)

print("📊 Model Performance Comparison:")
print(results_df)

best_model_name = results_df.iloc[0]['Model']
print(f"\n🏆 Best Model: {best_model_name}")


    # Select best model 
if r2_score(y_test, y_pred_lr) > r2_score(y_test, y_pred_gb):
        best_model = lr_grid.best_estimator_
        best_name = 'linear_regression'
        best_score = r2_score(y_test, y_pred_rf)
else:
        best_model = gb_grid.best_estimator_
        best_name = 'gradient_boosting'
        best_score = r2_score(y_test, y_pred_gb)
print(f"\nSelected best model: {best_name} with R² = {best_score:.4f}")

# Save & Export Model
import joblib

lr_grid.fit(X_train, y_train)

# check existence
predictions = lr_pipeline.predict(X_test)
print("Best estimator available:", hasattr(lr_grid, "best_estimator_"))

# then dump
joblib.dump(lr_grid.best_estimator_, 'crop_yield_model.pkl')
print("Model saved to crop_yield_model.pkl")



    # Summary per crop 
df_summary = df_crop_yield.groupby('Crop')[['Actual_Yield','Predicted_Yield']].mean().reset_index()
print("\nMean Actual vs Predicted Yield per Crop:")
print(df_summary)

    # Plot comparison per crop 
df_melt = df_summary.melt(id_vars=['Crop'],
                              value_vars=['Actual_Yield','Predicted_Yield'],
                              var_name='Type',
                              value_name='Mean_Yield')
plt.figure(figsize=(12,6))
sns.barplot(data=df_melt, x='Crop', y='Mean_Yield', hue='Type', dodge=True)
plt.xticks(rotation=45, ha='right')
plt.title('Mean Actual vs Predicted Yield per Crop')
plt.ylabel('Mean Yield')
plt.xlabel('Crop')
plt.tight_layout()
plt.show()

    # Scatter plot actual vs predicted
plt.figure(figsize=(8,8))
plt.scatter(df_results['Actual_Yield'], df_results['Predicted_Yield'], alpha=0.6)
lims = [min(df_results['Actual_Yield'].min(), df_results['Predicted_Yield'].min()),
            max(df_results['Actual_Yield'].max(), df_results['Predicted_Yield'].max())]
plt.plot(lims, lims, 'r--')
plt.xlabel('Actual Yield')
plt.ylabel('Predicted Yield')
plt.title(f'Actual vs Predicted Yield ({best_name})')
plt.tight_layout()
plt.show()

if __name__ == '__main__':
    main()