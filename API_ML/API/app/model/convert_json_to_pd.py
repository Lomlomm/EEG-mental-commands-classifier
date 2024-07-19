import pandas as pd
import requests

def ConvertTags2Float(df):
    label_mapping = {'open_hand': 0, 'close_hand': 1, 'rest': 2}
    df['classification'] = df['classification'].map(label_mapping)

    # Convertir los valores numéricos al tipo de dato float
    df['classification'] = df['classification'].astype(float)

    return df


def Convert2DF(url:str):
    
    response = requests.get(url)
    data = response.json()
    response_data = data['response']
    df = pd.json_normalize(response_data)
    all_data = []
    for column in df.columns: 
        all_data.append( pd.json_normalize(df[column].T ))

    join_data = []

    for i, data in enumerate(all_data): 
        for row in all_data[i]:
            join_data.append(pd.json_normalize(all_data[i][row])) 
    
    joined_df = pd.concat(join_data, ignore_index=True)

    return joined_df
