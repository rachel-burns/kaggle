#data import and preparation
import pandas as pd

def get_data(filepath):
    return pd.read_csv(filepath)