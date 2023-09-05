import json
import pandas as pd
import dill
import os
from datetime import datetime
from pydantic import BaseModel
from os import walk

path = os.environ.get('PROJECT_PATH', '..')
filenames = next(walk('../data/test'), (None, None, []))[2]

mod = sorted(os.listdir(f'{path}/data/models/'))[0]
with open(f'{path}/data/models/{mod}', 'rb') as file:
    model = dill.load(file)

class Form(BaseModel):
    description: str
    fuel: str
    id: int
    image_url: str
    lat: float
    long: float
    manufacturer: str
    model: str
    odometer: float
    posting_date: str
    price: int
    region: str
    region_url: str
    state: str
    title_status: str
    transmission: str
    url: str
    year: float


class Prediction(BaseModel):
    id: str
    price: int
    price_category: str

def predict():
    df_pred = pd.DataFrame()
    for filename in filenames:
        filename = '../data/test/' + filename
        with open(filename) as data_file:
            data = json.load(data_file)
        df = pd.json_normalize(data)
        y = model.predict(df)
        df_append = pd.Series(y)
        df_pred = pd.concat([df_pred, df_append])
    prediction_filename = f'{path}/data/predictions/predictions_{datetime.now().strftime("%Y%m%d%H%M")}.csv'
    df_pred.to_csv(prediction_filename, index=False)
    return 'predictions downloaded to file predictions.csv'


if __name__ == '__main__':
    predict()
