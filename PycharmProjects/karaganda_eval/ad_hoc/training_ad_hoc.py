import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
import mlflow
import optuna

def evaluate_prediction(y_test: pd.Series, y_pred: pd.Series) -> float:
    # Ensure the indices are aligned for the difference calculation
    y_test = y_test.reset_index(drop=True)
    y_pred = pd.Series(y_pred, name='y_pred').reset_index(drop=True)

    # Calculate the absolute difference and its percentage w.r.t y_test
    diff = np.abs(y_test - y_pred)
    diff_percentage = diff / y_test

    # Prepare a DataFrame for neat visualization
    result_df = pd.concat(
        [y_test.rename('y_test'), y_pred, diff.rename('diff'), diff_percentage.rename('diff_percentage')], axis=1)

    # Print statistics
    print(result_df.describe())

    # Calculate the percentage of predictions with a difference greater than 10%
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

dat = pd.read_csv(r'C:\Users\Наргис\AppData\Local\Programs\Python\Python310\Scripts\karaganda.csv', encoding = 'cp1251', delimiter = ';')

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    pd.set_option('display.max_columns', None)
    df = df.drop(columns=['Улица','Дата последнего осмотра','Дата вложение файла специалистом банка','Причины перерассмотрения','Стоимость по ценовой зоне за единицу','Стоимость по ценовой зоне','Стоимость по модели крыши','Стоимость по модели крыши за единицу:','Отклонение','Целевое назначение земельного участка'])
    df = df.dropna(subset=['Сумма сделки'])
    df.dropna(subset=['Село/Перекресток/Улица'])
    df.drop(columns=['Подход оценки','Метод оценки'])
    df['Этаж']=df['Этаж'].astype(int)
    df['Сумма сделки'] = df['Сумма сделки'].str.replace(',', '.').astype(float).astype(int)
    df['Общая площадь'] = df['Общая площадь'].str.replace(',', '.').astype(float).astype(int)
    repl = {
    'Кирпич кирпичные': 'Кирпич',
    'Панель': 'Ж/б панели',
    'Газоблок облицовка кирпичом': 'Газоблок',
    'Газоблок обл. кирпич': 'Газоблок',
    'Газоблок облицовка кирпичом': 'Газоблок',
    'Монолит бетон':'Монолит',
    'Газоблок обложенные кирпичом': 'Газоблок',
    'Железобетон':'Ж/б панели',
    'Пескоблок СКЦ':'Шлакоблок',
    'Пеноблочный монолитный бетон., пеноблочные':'Пеноблочный',
    'Ж/б блок':'Ж/б панели',
    'Ж/б панели ж/б панель':'Ж/б панели',
    'Пенобетон пенобетонные блоки':'Пеноблочный',
    'Монолит пеноблоки':'Монолит пеноблок'
    }

    df['Материал стен'] = df['Материал стен'].str.strip().replace(repl)
    return df

preprocess_data(dat)

client = mlflow.tracking.MlflowClient()

start_dates =  ['2022-01-01',
               '2023-01-01']
dependent_vars = ['Сумма сделки']

features_common = ['Общая площадь', 'Этажность', 'Этаж', 'Год постройки',
                   'Материал стен', 'Рыночная стоимость', 'Рыночная стоимость за кв. м',
                   'Широта', 'Долгота']



for dep_var in dependent_vars:
    experiment_name = f"test_{dep_var}"
    mlflow.set_experiment(experiment_name)

    features = features_common

    for start_date in start_dates:
        # Filtering data
        current_data = dat[dat['Дата сделки'] >= start_date]
        test_mask = current_data['Дата сделки'] >= '2023-09-15'
        current_data = current_data[features + [dep_var]]  # Only select columns we need

        train = current_data[~test_mask]
        test = current_data[test_mask]

        # Initialize encoders
        label_encoder = LabelEncoder()
        ordinal_encoder = OrdinalEncoder()

        # Create new DataFrames
        train_tree = train.copy()
        test_tree = test.copy()

        # Label Encoding for WallMaterialGroupName
        train_tree['WallMatEnc'] = label_encoder.fit_transform(train_tree['Материал стен'])
        test_tree['WallMatEnc'] = label_encoder.transform(test_tree['Материал стен'])


        # Drop the original columns from train_tree and test_tree
        train_tree = train_tree.drop(columns=['Материал стен'])
        test_tree = test_tree.drop(columns=['Материал стен'])

        cat_features = ['Материал стен']

        models = {
            'RandomForest': RandomForestRegressor(),
            'XGBoost': xgb.XGBRegressor(),
            'LightGBM': lgb.LGBMRegressor(),
            'CatBoost': CatBoostRegressor(cat_features=cat_features, verbose=0)
        }

        for name, model in models.items():
            run_name = f"{name}_{start_date}"
            with mlflow.start_run(run_name=run_name):
                # Log model, algorithm, and start date
                mlflow.set_tag("start_date", start_date)
                mlflow.log_params({
                    "model": name,
                    "algorithm": name,
                    "start_date": start_date
                })

                if name == 'CatBoost':
                    model.fit(train.drop(dep_var, axis=1), train[dep_var])
                    y_pred = model.predict(test.drop(dep_var, axis=1))
                else:
                    model.fit(train_tree.drop(dep_var, axis=1), train_tree[dep_var])
                    y_pred = model.predict(test_tree.drop(dep_var, axis=1))

                # Log model's hyperparameters
                if hasattr(model, "get_params"):
                    mlflow.log_params(model.get_params())

                metrics = compute_metrics(test_tree[dep_var], y_pred)
                mlflow.log_metrics(metrics)

import re


def extract_floor(value):
    match = re.search(r'\d+', str(value))
    if match:
        return int(match.group())
    else:
        return None


dat['Этажность'] = dat['Этажность'].apply(extract_floor)

cat_features = ['Материал стен']

df = dat[dat['Дата сделки'] >= '2022-01-01']
X = df[features_common]
y = df['Сумма сделки']

X_train = X[~test_mask]
X_test = X[test_mask]
y_train = y[~test_mask]
y_test = y[test_mask]

catboost = CatBoostRegressor(cat_features=cat_features, verbose=0)
catboost.fit(X_train, y_train)
y_pred = catboost.predict(X_test)
evaluate_prediction(y_test, y_pred)

def objective(trial):
    # Hyperparameter suggestions
    params = {
        'iterations': trial.suggest_int('iterations', 50, 300),
        'depth': trial.suggest_int('depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'random_strength': trial.suggest_float('random_strength', 1, 10),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.1, 1.0),
        'od_type': trial.suggest_categorical('od_type', ['IncToDec', 'Iter']),
        'od_wait': trial.suggest_int('od_wait', 10, 50)
    }

    model = CatBoostRegressor(**params, cat_features=cat_features, verbose=0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return evaluate_prediction(y_test, y_pred)  # Assuming this returns a single scalar metric

# Optuna studies
study = optuna.create_study(direction="minimize")  # Assuming you're minimizing your metric
study.optimize(objective, n_trials=100)


# Best parameters for both models
best_params = study.best_params

print(f"Best parameters for non-adjusted model: {best_params_no_adj}")

# Using best parameters for the non-adjusted model
catboost_model = CatBoostRegressor(cat_features=cat_features, verbose=0, **best_params)
catboost_model.fit(X_train, y_train)
y_pred = catboost_model.predict(X_test)
evaluate_prediction(y_test, y_pred)

y_test_series = pd.Series(y_test, name="Стоимость без поправки на инфляцию")
y_pred_series = pd.Series(y_pred, name="Предсказание стоимость без поправки на инфляцию")

# Reset index for X_test_adj to make sure it aligns with the new Series
X_test_reset = X_test_no_adj.reset_index(drop=True)
# Convert numpy arrays to pandas Series and reset their indices
y_test_no_adj_series = pd.Series(y_test_no_adj, name="Стоимость без поправки на инфляцию").reset_index(drop=True)
y_pred_no_adj_series = pd.Series(y_pred_no_adj, name="Предсказание стоимость без поправки на инфляцию").reset_index(drop=True)

# Concatenate with the reset-index DataFrame
new_df = pd.concat([X_test_no_adj_reset, y_test_adj_series, y_pred_adj_series, y_test_no_adj_series, y_pred_no_adj_series], axis=1)
new_df.to_excel('karg/predictions.xlsx', index=False)




