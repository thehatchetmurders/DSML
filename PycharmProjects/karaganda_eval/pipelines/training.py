import os

import pandas as pd
import numpy as np
import mlflow
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from mlflow.tracking import MlflowClient
import datetime
import optuna
import pickle
import mlflow.pyfunc
import matplotlib.pyplot as plt
from catboost import Pool
from catboost import CatBoostRegressor
from mlflow.models import infer_signature
from mlflow.exceptions import MlflowException
import re
from dateutil import parser


def evaluate_prediction(y_test: pd.Series, y_pred: pd.Series) -> float:
    y_test = y_test.reset_index(drop=True)
    y_pred = pd.Series(y_pred, name='y_pred').reset_index(drop=True)

    diff = np.abs(y_test - y_pred)
    diff_percentage = diff / y_test

    result_df = pd.concat(
        [y_test.rename('y_test'), y_pred, diff.rename('diff'), diff_percentage.rename('diff_percentage')], axis=1)

    print(result_df.describe())

    percentage_diff = (diff > 0.1 * y_test).mean() * 100
    print(f"The percentage of y_pred values differing by more than 10% from y_test is: {percentage_diff:.2f}%")

    return percentage_diff


def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    rlmse = np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred)))
    r2 = r2_score(y_true, y_pred)
    custom_metric = evaluate_prediction(y_true, y_pred)
    return {
        'MAE': mae,
        'RMSE': rmse,
        'RLMSE': rlmse,
        'R2': r2,
        'Custom Metric': custom_metric
    }


dat = pd.read_csv(r'C:\Users\Наргис\AppData\Local\Programs\Python\Python310\Scripts\karaganda.csv', encoding='cp1251',
                  delimiter=';')
pd.set_option('display.max_columns', None)
dat = dat.drop(
    columns=['Улица', 'Дата последнего осмотра', 'Дата вложение файла специалистом банка', 'Причины перерассмотрения',
             'Стоимость по ценовой зоне за единицу', 'Стоимость по ценовой зоне', 'Стоимость по модели крыши',
             'Стоимость по модели крыши за единицу:', 'Отклонение', 'Целевое назначение земельного участка'])
dat = dat.dropna(subset=['Сумма сделки'])
dat.dropna(subset=['Село/Перекресток/Улица'])
dat.drop(columns=['Подход оценки', 'Метод оценки'])
dat['Этаж'] = dat['Этаж'].astype(int)
dat['Сумма сделки'] = dat['Сумма сделки'].str.replace(',', '.').astype(float).astype(int)
dat['Общая площадь'] = dat['Общая площадь'].str.replace(',', '.').astype(float).astype(int)


def extract_floor(value):
    match = re.search(r'\d+', str(value))
    if match:
        return int(match.group())
    else:
        return None


dat['Этажность'] = dat['Этажность'].apply(extract_floor)

start_dates = ['2022-01-01',
               '2023-01-01']
y = dat['Сумма сделки']

X = dat[['Общая площадь', 'Этажность', 'Этаж', 'Год постройки',
         'Материал стен', 'Рыночная стоимость', 'Рыночная стоимость за кв. м',
         'Широта', 'Долгота', 'Дата сделки']]

cat_features = dat['Материал стен']

test_mask = dat['Дата сделки'] >= '2023-09-20'
X_train, y_train = X[~test_mask], y[~test_mask]
X_test, y_test = X[test_mask], y[test_mask]

client = mlflow.tracking.MlflowClient()
mlflow.get_tracking_uri()

repl = {
    'Кирпич кирпичные': 'Кирпич',
    'Панель': 'Ж/б панели',
    'Газоблок облицовка кирпичом': 'Газоблок',
    'Газоблок обл. кирпич': 'Газоблок',
    'Газоблок облицовка кирпичом': 'Газоблок',
    'Монолит бетон': 'Монолит',
    'Газоблок обложенные кирпичом': 'Газоблок',
    'Железобетон': 'Ж/б панели',
    'Пескоблок СКЦ': 'Шлакоблок',
    'Пеноблочный монолитный бетон., пеноблочные': 'Пеноблочный',
    'Ж/б блок': 'Ж/б панели',
    'Ж/б панели ж/б панель': 'Ж/б панели',
    'Пенобетон пенобетонные блоки': 'Пеноблочный',
    'Монолит пеноблоки': 'Монолит пеноблок'
}

dat['Материал стен'] = dat['Материал стен'].str.strip().replace(repl)

cat_features = ['Материал стен']

X_train, y_train = X[~test_mask], y[~test_mask]
X_test, y_test = X[test_mask], y[test_mask]

mlflow.set_tracking_uri('file:///C:/Users/Наргис/Desktop/mlruns')

experiment_name = "karg_house_prediction_" + datetime.datetime.today().strftime('%Y-%m-%d')
mlflow.set_experiment(experiment_name)

model = CatBoostRegressor(iterations=200, depth=7, learning_rate=0.08, loss_function='RMSE', verbose=200)

dat['Timestamp'] = dat['Дата сделки'].apply(lambda x: parser.parse(x).timestamp())

dat['Дата сделки'] = dat['Дата сделки'].str.replace(' 00:00:00', '')

start_dates = ['2022-01-01',
               '2023-01-01']
y = dat['Сумма сделки']

X = dat[['Общая площадь', 'Этажность', 'Этаж', 'Год постройки',
         'Материал стен', 'Широта', 'Долгота']]

cat_features = dat['Материал стен']

test_mask = dat['Timestamp'] >= 1692672000.0
X_train, y_train = X[~test_mask], y[~test_mask]
X_test, y_test = X[test_mask], y[test_mask]

cat_features = ['Материал стен']
current_date_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
algo_name = "CatBoost"

with mlflow.start_run(run_name=f"{current_date_time}_{algo_name}"):
    model.fit(X_train, y_train, cat_features=cat_features, eval_set=(X_test, y_test), early_stopping_rounds=30)

    predictions = model.predict(X_test)
    metrics = compute_metrics(y_test, predictions)

    mlflow.log_param("features", ', '.join(X.columns))
    mlflow.log_param("algorithm", "CatBoost")

    mlflow.log_param("train_start_date", dat['Timestamp'][~test_mask].min())
    mlflow.log_param("test_start_date", dat['Timestamp'][test_mask].min())

    mlflow.log_param("num_rows_train", len(X_train))
    mlflow.log_param("num_rows_test", len(X_test))
    mlflow.log_param("categorical_features", ', '.join(cat_features))

    for metric_name, metric_value in metrics.items():
        mlflow.log_metric(metric_name, metric_value)

        feature_importances = model.get_feature_importance(Pool(X_test, label=y_test, cat_features=cat_features))
        feature_names = X.columns
        for feature, importance in zip(feature_names, feature_importances):
            mlflow.log_metric(f"feature_importance_{feature}", importance)

        signature = mlflow.models.infer_signature(X_train, predictions)
        mlflow.catboost.log_model(model, "model", signature=signature)

feature_importances = model.get_feature_importance(Pool(X_test, label=y_test, cat_features=cat_features))
sorted_feature_importance = feature_importances.argsort()
plt.barh([feature_names[i] for i in sorted_feature_importance], feature_importances[sorted_feature_importance])
plt.xlabel("CatBoost Feature Importance")
plt.show()
# Select the top features based on cumulative importance
feature_importances_normalized = feature_importances / np.sum(feature_importances)

# Sort the feature importances
sorted_idx = np.argsort(feature_importances_normalized)[::-1]

# Compute the cumulative feature importance
cumulative_importance = np.cumsum(feature_importances_normalized[sorted_idx])

# Determine the number of features required to reach the threshold
# For example, if you want the features that account for 95% of the importance
threshold = 0.95
important_feature_count = np.where(cumulative_importance > threshold)[0][0] + 1

selected_features = X_train.columns[sorted_idx[:important_feature_count]]
selected_features

X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]
cat_features_selected = X_train_selected.nunique()[X_train_selected.nunique() < 10].keys().tolist()
cat_features_selected


def objective(trial):
    # Suggest hyperparameters
    depth = trial.suggest_int("depth", 6, 10)
    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.1, log=True)
    iterations = trial.suggest_int("iterations", 100, 1000, step=100)
    l2_leaf_reg = trial.suggest_float("l2_leaf_reg", 1e-5, 10, log=True)
    border_count = trial.suggest_int("border_count", 32, 255, step=8)
    bagging_temperature = trial.suggest_float("bagging_temperature", 0, 1)

    # Train model using the suggested hyperparameters
    model = CatBoostRegressor(
        iterations=iterations,
        depth=depth,
        learning_rate=learning_rate,
        l2_leaf_reg=l2_leaf_reg,
        border_count=border_count,
        bagging_temperature=bagging_temperature,
        loss_function='RMSE',
        verbose=200
    )
    model.fit(X_train_selected, y_train, cat_features=cat_features_selected, eval_set=(X_test_selected, y_test),
              early_stopping_rounds=30)

    # Make predictions and compute RMSE
    predictions = model.predict(X_test_selected)
    rmse = mean_squared_error(y_test, predictions, squared=False)

    return rmse


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=10)

best_trial = study.best_trial

best_model = CatBoostRegressor(
    iterations=best_trial.params['iterations'],
    depth=best_trial.params['depth'],
    learning_rate=best_trial.params['learning_rate'],
    l2_leaf_reg=best_trial.params['l2_leaf_reg'],
    border_count=best_trial.params['border_count'],
    bagging_temperature=best_trial.params['bagging_temperature'],
    loss_function='RMSE',
    verbose=200
)

with mlflow.start_run(run_name=f"{algo_name}_registered_model"):
    # Train the best model
    best_model.fit(X_train_selected, y_train, cat_features=cat_features_selected, eval_set=(X_test_selected, y_test),
                   early_stopping_rounds=30)

    # Make predictions and compute metrics
    predictions = best_model.predict(X_test_selected)
    metrics = compute_metrics(y_test, predictions)

    # add input for model
    sample_input = X_train_selected.head()

    # Infer the model signature
    signature = infer_signature(sample_input, predictions)

    # Log the best parameters and metrics
    mlflow.log_params(best_trial.params)

    # Log other parameters...
    # Log static parameters, those not optimized by Optuna
    mlflow.log_param("features", ', '.join(X_train_selected.columns))
    mlflow.log_param("algorithm", "CatBoost")
    mlflow.log_param("loss_function", 'RMSE')
    mlflow.log_param("verbose", 200)
    mlflow.log_param("train_start_date", dat['Timestamp'][~test_mask].min())
    mlflow.log_param("test_start_date", dat['Timestamp'][test_mask].min())
    mlflow.log_param("num_rows_train", len(X_train_selected))
    mlflow.log_param("num_rows_test", len(X_test_selected))
    mlflow.log_param("categorical_features", ', '.join(cat_features_selected))

    tags = {
        "developer": "your_name",
        "algorithm": "CatBoost",
        "project": "Karaganda",
        "run_date": datetime.datetime.now().strftime("%Y-%m-%d"),
        "purpose": "Testing different parameters",
        "data_split": "time_series",
        "model_type": "best_model",
        "target variable": "Deal_Price_per_1_m2",
        # ... any other tags you want to include
    }
    mlflow.set_tags(tags)

from mlflow.tracking import MlflowClient

client = MlflowClient()

# registered_model = client.get_registered_model(name=f"{algo_name}_registered_model")

for metric_name, metric_value in metrics.items():
    mlflow.log_metric(metric_name, metric_value)

    # Log feature importance
feature_importance = best_model.get_feature_importance()
for feature, importance in zip(X.columns, feature_importance):
    mlflow.log_metric(f"feature_importance_{feature}", importance)

# Log the model with the CatBoost flavor
mlflow.catboost.log_model(best_model, "model", signature=signature)
model_uri = os.getcwd() + "\mlruns\models"
registered_model_name = "Kar_catboost_2024_Feb"

result = mlflow.register_model(model_uri, registered_model_name)

for tag_key, tag_value in tags.items():
    client.set_model_version_tag(
        name=registered_model_name,
        version=result.version,
        key=tag_key,
        value=tag_value
    )

best_model.save_model(os.getcwd() + "\mlruns\Kar_catboost_2024_Feb.cbm", format='cbm')

model_name = "Kar_catboost_2024_Feb"
model_version = 3  # replace with the correct version number
loaded_model = mlflow.catboost.load_model(os.getcwd()+f"/mlruns/models:/{model_name}/{model_version}")


