import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics.pairwise import haversine_distances
from math import radians
import mlflow.pyfunc
import pickle


def find_project_root(current_dir):
    while not os.path.exists(os.path.join(current_dir, '.project_root')):
        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:
            raise Exception("Project root not found.")
        current_dir = parent_dir
    return current_dir

project_root = find_project_root(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(project_root, 'data')

print("Project root directory:", project_root)
print("Data directory:", DATA_DIR)

pipelines_dir = os.path.join(project_root, 'src', 'pipelines')
if pipelines_dir not in sys.path:
    sys.path.append(pipelines_dir)

print("Project root directory:", project_root)
print("Data directory:", DATA_DIR)

from scraping_stat_gov import process

model_name = "Kar_catboost_2024_Feb.cbm"
model_version = 0  # replace with the correct version number
loaded_model = mlflow.catboost.load_model(r"C:\Users\Наргис\PycharmProjects\karaganda_eval\pipelines")
loaded_model
#loaded_model.save_model("Almaty_catboost_2023_Nov_version_4.cbm", format="cbm")
export_path = r"C:\Users\Наргис\Desktop\models11"


# Save the model
# mlflow.pyfunc.save_model(path=export_path, python_model=loaded_model)
#
# feature_names = loaded_model.feature_names_

