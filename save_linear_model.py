import joblib

# You must define and instantiate lr_pipeline and lr_param_grid before this
lr_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

lr_param_grid = {
    'regressor__fit_intercept': [True, False],
    'regressor__positive': [True, False]
}

lr_grid = GridSearchCV(estimator=lr_pipeline,
                       param_grid=lr_param_grid,
                       cv=5)

# Fit the grid search object
lr_grid.fit(X_train, y_train)

# Check that the best estimator exists
print("Best estimator available:", hasattr(lr_grid, "best_estimator_"))

# Use the best estimator for predictions
best_model = lr_grid.best_estimator_
y_pred = best_model.predict(X_test)

# … compute metrics for y_pred vs y_test here …

# Save the best model (with joblib)
joblib.dump(best_model, 'crop_yield_model.pkl')
print("Model saved to crop_yield_model.pkl")