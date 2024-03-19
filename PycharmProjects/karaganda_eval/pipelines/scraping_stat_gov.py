import pandas as pd
import numpy as np
import requests
import os

DATA_URL = "https://taldau.stat.gov.kz/ru/Api/GetIndexData/703083?period=4&dics=67,848,2817"
from config import DATA_DIR
OUTPUT_FILE_NAME = 'inflation_apartment.csv'

OUTPUT_PATH = os.path.join(DATA_DIR, OUTPUT_FILE_NAME)

def fetch_data(url: str) -> pd.DataFrame:
    """Fetch data from the given URL and return as a DataFrame."""
    response = requests.get(url)
    return pd.DataFrame(response.json())

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the fetched data."""

    periods_df = df.explode('periods').reset_index(drop=True)
    periods_details = pd.json_normalize(periods_df['periods'])

    result = pd.concat([
        periods_df.drop(columns=['periods', 'terms', 'termNames']),
        periods_df['terms'].apply(pd.Series).reset_index(drop=True),
        periods_df['termNames'].apply(pd.Series).reset_index(drop=True),
        periods_details
    ], axis=1)

    result.columns = ['Original_Column1', 'Term1', 'Term2', 'region', 'period_type', 'inflation_indicator',
                      'period_name', 'period_date', 'period_value']
    result = result.sort_values(by=['period_value', 'period_date'], ascending=[False, False])
    result['period_value'] = result['period_value'].replace('x', np.NaN)
    result['period_value'] = result['period_value'].astype(float)
    result['period_date'] = pd.to_datetime(result['period_date'])
    result = result[(result['region'] == 'Г.АЛМАТЫ') | (result['region'] == 'РЕСПУБЛИКА КАЗАХСТАН') | (
                result['region'] == 'Г.АСТАНА')]
    translation_map = {
        "РЕСПУБЛИКА КАЗАХСТАН": "kazakhstan",
        "Г.АЛМАТЫ": "almaty",
        "Г.АСТАНА": "astana"
    }
    result['region'] = result['region'].map(translation_map).str.strip().str.lower()

    result['period_date'] = pd.to_datetime(result['period_date'], dayfirst=True)  # Addressing the UserWarning
    result = result[(result['inflation_indicator'] == 'Арендная плата за благоустроенное жилье') |
                    (result[
                         'inflation_indicator'] == 'Перепродажа благоустроенного жилья (квартиры в многоквартирных домах)') |
                    (result['inflation_indicator'] == 'Продажа нового жилья (квартиры в многоквартирных домах)')]
    result['period_value'] = result['period_value'].replace('x', np.NaN)
    result['period_value'] = result['period_value'].astype(float)

    result = result[(result['period_type'] != '77212653')]
    result = result[(result['inflation_indicator'] == 'Продажа нового жилья (квартиры в многоквартирных домах)')]
    result['inflation_indicator'] = result['inflation_indicator'].replace(
        'Продажа нового жилья (квартиры в многоквартирных домах)',
        'sale_of_new_housing_(apartments_in_multi-storey_buildings)'
    ).str.lower().str.replace(' ', '_')
    result = result[result['period_type'] == 'отчетный период к предыдущему периоду']
    result['period_type'] = result['period_type'].replace(
        'отчетный период к предыдущему периоду',
        'reporting_period_to_the_previous_period'
    ).str.lower().str.replace(' ', '_')
    result['inflation'] = result['region'] + '_' + result['period_type'] + '_' + result['inflation_indicator']

    inflation_apartment = pd.pivot_table(result, values='period_value', index=['period_date'], columns=['inflation'])

    return inflation_apartment

def save_data(df: pd.DataFrame, file_path: str) -> None:
    """Save the preprocessed data to an excel file."""
    # Ensure the data directory exists
    data_dir = os.path.dirname(file_path)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    df.to_csv(file_path)

def process() -> None:
    print(f"Current Working Directory: {os.getcwd()}")
    print("Fetching data...")
    raw_data = fetch_data(DATA_URL)

    print("Preprocessing data...")
    processed_data = preprocess_data(raw_data)

    print(f"Saving processed data to {OUTPUT_PATH}...")
    print(OUTPUT_PATH)
    save_data(processed_data, OUTPUT_PATH)

    print("Processing completed.")

# Other functions or operations specific to this module...

if __name__ == "__main__":
    process()
