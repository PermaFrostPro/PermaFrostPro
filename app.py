import pandas as pd 
import matplotlib.pyplot as plt
import eel
import plotly.express as px
from io import StringIO
from geopy import Point

import xgboost as xgb
import joblib

model = joblib.load('xgb.joblib')

def get_lat_lon(coord_str):
    lat_str, lon_str = coord_str.split()
    lat_dir, lon_dir = lat_str[-1], lon_str[-1]
    lat_value, lon_value = map(float, (lat_str[:-2], lon_str[:-2]))
    lat = lat_value if lat_dir == 'N' else -lat_value
    lon = lon_value if lon_dir == 'E' else -lon_value
    return lat, lon

@eel.expose
def give_res(csv_file: str) -> str:
    df = pd.read_csv(StringIO(csv_file), delimiter=',')
    coords = df['координаты'].apply(get_lat_lon)
    df[['широта', 'долгота']] = pd.DataFrame(coords.tolist())
    df['взаимосвязь с водой бин'] = df['взаимосвязь с водой'].apply(lambda x: 1 if x == 'да' else 0)
    for elem in ['болото', 'бугор пучения', 'лес', 'озеро', 'торфяник']:
        df[elem] = df['ландшафт'].apply(lambda x: 1 if x == elem else 0)
    df['взаимосвязь с водой'] = df['взаимосвязь с водой бин'] 
    coord_df = df[['широта', 'долгота']]
    df = df.drop(columns=['широта','долгота','ландшафт', 'взаимосвязь с водой бин','координаты'])
    df['температура воздуха'] = [x if isinstance(x, float) else 0 for x in df['температура воздуха']]
    df['торф'] = [x if isinstance(x, float) or  isinstance(x, int) else 0 for x in df['торф']]
    y_pred = model.predict(df)
    coord_df['Z'] = y_pred
    color_map = {0: '#34eb52', 1: '#ebd934', 2: '#eb4034'}
    types_map = {0: 'нет мерзлоты', 1: 'несливающая мерзлота', 2: 'сливающаяся мерзлота'}

    coord_df['color'] = coord_df['Z'].map(color_map)
    coord_df['name'] = coord_df['Z'].map(types_map)
    coord_df['size']=10

    fig = px.scatter_mapbox(coord_df, lat='широта', lon='долгота', hover_name='name', zoom=12, height=600,color='color',size='size')
    fig.update_layout(mapbox_style='open-street-map', showlegend=False)
    
    fig.write_html('web/map.html')
    return 'map.html'

eel.init('web')
eel.start('main.html', size=(1050,1050))