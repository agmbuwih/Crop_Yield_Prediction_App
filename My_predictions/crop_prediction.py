import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.pyplot as plt

data = pd.read_csv('Crop_Yield_Prediction.csv')
data

data.head()

data.tail()

data.info()

data.duplicated().sum()

data.describe()

missing_values = data.isnull().sum()
missing_values

data.nunique()

data['Crop'].unique()

crop_average = pd.pivot_table(data, index=['Crop'], aggfunc='mean')
crop_average

crop_average_new = crop_average.reset_index()
crop_average_new

data1 = data[['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'pH_Value', 'Rainfall']]

for i in data1.columns:
    plt.figure(figsize = (15, 6))
    sns.barplot(
    x='Crop',
    y=i,
    hue='Crop',             
    data=crop_average_new,
    palette='mako',
    legend=False            
)    
    
data1.corr()
fig, ax = plt.subplots(1, 1, figsize = (10,8))
sns.heatmap(data1.corr(), annot = True, cmap = 'coolwarm')
ax.set(xlabel = 'Features')
ax.set(ylabel = 'Features')
plt.title('Correlation between different features', fontsize=15, color='black', loc='center')
plt.show()
# displays the heatmap plot.

# Regression pipeline for crop yields

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.pipeline import make_pipeline
# from sklearn import set_config
from sklearn.preprocessing import StandardScaler

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder,MinMaxScaler

#regressionmodels
#from sklearn.linear_model import SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

#metrics for measuring performance
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, mean_absolute_percentage_error, r2_score

data = pd.read_csv('Crop_Yield_Prediction.csv')
data

data['Crop'].unique()

df = pd.pivot_table(data, index=['Crop'], aggfunc='mean')
df

data.head()

X = data.drop('Yield', axis=1)
y = data['Yield']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train_model = X_train, y_train

model_scores = []
def score_model(model_name, y_true, y_pred):
  scores = {
      'Model': model_name,
      'MAE ()': round(mean_absolute_error(y_true, y_pred), 2),
      'RMSE ()': round(root_mean_squared_error(y_true, y_pred), 2),
      'MAPE (%)': round(100 * mean_absolute_percentage_error(y_true, y_pred), 2),
      'R-Squared': round(r2_score(y_true, y_pred), 3)
  }
  return scores

data.select_dtypes(include='number').columns

sns.relplot(data = data,
              x='Rainfall',
              y='Yield',
              height=5,
              aspect=1.2)

baseline_pred = y_train.mean()
baseline_preds = pd.Series([baseline_pred] * len(y_test))
baseline_preds

model_scores.append(score_model('dummy', y_test, baseline_preds))
pd.DataFrame(model_scores)

# Select categorical and numeric columns
X_cat = X.select_dtypes(exclude="number").copy()
X_num = X.select_dtypes(include="number").copy()

# Numeric pipeline to impute missing values
numeric_pipe = make_pipeline(
    SimpleImputer(strategy="mean"),
    StandardScaler()
)
# Categoric pipeline (OneHot)
categoric_pipe = make_pipeline(
    SimpleImputer(strategy="constant", fill_value="N_A"),
    OneHotEncoder(handle_unknown="ignore", sparse_output=False)
)

# now put everything together in a columntransformer
preprocessor = make_column_transformer(
    # each step is passed in a tuple seperated by comma ;)
    (numeric_pipe, X_num.columns),
    (categoric_pipe, X_cat.columns)
)

preprocessor

dt_pipe = make_pipeline(preprocessor,
                        DecisionTreeRegressor())
dt_pipe

dt_pipe.fit(X_train, y_train)

dt_predictions = dt_pipe.predict(X_test)

model_scores.append(score_model('decision tree', y_test, dt_predictions))
pd.DataFrame(model_scores)

knn_pipe = make_pipeline(preprocessor,
                         KNeighborsRegressor())
knn_pipe.fit(X_train, y_train)

knn_predictions = dt_pipe.predict(X_test)

model_scores.append(score_model('KNN_scaled', y_test, knn_predictions))
pd.DataFrame(model_scores)

rf_pipe = make_pipeline(preprocessor,
                        RandomForestRegressor())
rf_pipe.fit(X_train, y_train)

rf_preds = rf_pipe.predict(X_test)

model_scores.append(score_model('rf_scaled', y_test, rf_preds))
pd.DataFrame(model_scores)

reg_pipe = make_pipeline(preprocessor,
                      GradientBoostingRegressor())
reg_pipe.fit(X_train, y_train)

reg_preds = reg_pipe.predict(X_test)

model_scores.append(score_model('reg_scaled', y_test, reg_preds))
pd.DataFrame(model_scores)

lr_pipe = make_pipeline(preprocessor,
                     LinearRegression())
lr_pipe.fit(X_train, y_train)

lr_preds = lr_pipe.predict(X_test)

model_scores.append(score_model('lr_scaled', y_test, lr_preds))

pd.DataFrame(model_scores)


lr_pipe.get_params().keys()

param_grid = {
    'linearregression__fit_intercept': [True, False],
    'linearregression__positive': [True, False]
}

pipeline = make_pipeline(preprocessor,
                         LinearRegression())
pipeline

lr_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1
)
lr_search.fit(X_train, y_train)

lr_preds = lr_search.predict(X_test)

model_scores.append(score_model('lr_tuned', y_test, lr_preds))
pd.DataFrame(model_scores)

predictions = lr_pipe.predict(X_test)
predictions

Crop_column = data['Crop']
Crop_column

crop_test = Crop_column[:len(predictions)]
results = pd.DataFrame({'Crops': crop_test, 'Yields': predictions})
results

N = len(predictions)
results = pd.DataFrame({
    'Crop': Crop_column.iloc[:N].values,    # or use appropriate indexing
    'Yield': predictions
})
results

results['Yield'].unique()

results = results.sort_values(by='Yield', ascending=False)
results

from sklearn.metrics import root_mean_squared_log_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler,RobustScaler
from sklearn.linear_model import LinearRegression, BayesianRidge, SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest, chi2, f_regression
from sklearn.linear_model import SGDRegressor

def score_models(X_train, y_train, X_test, y_test):
  model_scaler = []
  mae = []
  rmse = []
  mape =[]
  r2 = []
  rm2_log = []
  feature_selection = []

  scores_df = {
      "Model_Scaler":model_scaler,
      "Feature Selector": feature_selection,
      'MAE ($)': mae,
      'RMSE ($)': rmse,
      'MAPE (%)': mape,
      'R-Squared': r2,
      "RMea2_log": rm2_log
  }
  # list of necessry scalers
  scalers_list = [MinMaxScaler(), StandardScaler(),RobustScaler()]
  # list of models
  models_list = [LinearRegression(), BayesianRidge(), SGDRegressor(), KNeighborsRegressor(),
                 SVR(),DecisionTreeRegressor(), RandomForestRegressor(),GradientBoostingRegressor()]
  # selectors_list = ["passthrough", VarianceThreshold(threshold=0), VarianceThreshold(threshold=0.2)]
  feature_score = ["all", 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]

  for i in range(len(models_list)):
    for j in range(len(scalers_list)):
      # for k in range(len(selectors_list)):
        for o in range(len(feature_score)):
          pipeline = make_pipeline(
            preprocessor,
            scalers_list[j],
            # selectors_list[k],
            SelectKBest(score_func=f_regression, k=feature_score[o]),
            models_list[i]
          )
          pipeline.fit(X_train, y_train)
          # if (round(r2_score(y_test, pipeline.predict(X_test)), 2)) > 0.65:
          model_scaler.append(f"{models_list[i]} {scalers_list[j]}")
          feature_selection.append(feature_score[o])
          mae.append(str(round(mean_absolute_error(y_test, pipeline.predict(X_test)), 2)))
          rmse.append(str(round(root_mean_squared_error(y_test, pipeline.predict(X_test)), 2)))
          mape.append(str(round(100 * mean_absolute_percentage_error(y_test, pipeline.predict(X_test)), 2)))
          r2.append(str(round(r2_score(y_test, pipeline.predict(X_test)), 2))),
          rm2_log.append(round(root_mean_squared_log_error(y_test, pipeline.predict(X_test)), 3)),
  return  pd.DataFrame(scores_df)

score_model=score_models(X_train, y_train, X_test, y_test)

score_model.sort_values("RMea2_log").head()

df_res = score_model.sort_values("RMea2_log")

df_sorted = df_res.sort_values(by="RMSE ($)", ascending=True)
df_sorted

sgd = make_pipeline(preprocessor,
                            SGDRegressor())
sgd.fit(X_train, y_train)

predictions = sgd.predict(X_test)
predictions

Crop_column = data['Crop']
Crop_column

crop_test = Crop_column[:len(predictions)]
results = pd.DataFrame({'Crops': crop_test, 'Yields': predictions})
results

N = len(predictions)
results = pd.DataFrame({
    'Crop': Crop_column.iloc[:N].values,    # or use appropriate indexing
    'Yield': predictions
})
results

results['Yield'].unique()

results = results.sort_values(by='Yield', ascending=False)
results

from sklearn.pipeline import make_pipeline
from sklearn.linear_model import SGDClassifier
import joblib

lr = lr_pipe.predict(X_test)
model = joblib.dump(lr, 'model.joblib')
