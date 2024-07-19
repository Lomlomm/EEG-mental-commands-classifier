from pathlib import Path
import pandas as pd

def label_data(dataframe):
    num_rows = len(dataframe)
    left_rows = num_rows - 15360
    # Create classification labels for each row 
    column = ['right' for _ in range(10240)] + ['rest' for _ in range(5120)] + ['left' for _ in range(left_rows)]
    
    # Set the classification column values 
    dataframe['Classification'] = column
    return dataframe


def read_csvs(path:str):
    # Verificar la ruta completa
    full_path = Path(path).resolve()
    print("Full path:", full_path)

    # Verificar si hay archivos CSV en la carpeta
    csv_files = list(full_path.glob('*.csv'))

    dfs = [pd.read_csv(file) for file in csv_files]

    csv_files = list(full_path.glob('*.csv'))
    csv_files = list(csv_files)

    return [dfs, csv_files]

def process_row(row, iteration:int, label:str):
    del row['Time:512Hz']
    rename = {f'Channel {i}': f'CH{i}{label}' for i in range(1, 15)}
    row['Time:512Hz'] = iteration + 1 

    for key, value in rename.items(): 
        row[value] = row[key]
        del row[key]
    
    return row

def process_data(dataframes):
    labeled_dataframes_list = []
    drop_columns = ['Epoch', 'Event Id', 'Event Date', 'Event Duration', 'Channel 15', 'Channel 16']

    for i in range(len(dataframes[0])):
        df = dataframes[0][i].copy()
        df.drop(columns = drop_columns, inplace=True)

        dataframe_labeled = label_data(df)
        labeled_dataframes_list.append(dataframe_labeled)

        path = dataframes[1][i] 
        if not Path(path.parent / 'Processed').exists():
            Path(path.parent / 'Processed').mkdir()

        dataframe_labeled.to_csv(path.parent / 'Processed' / path.name, index=False)
    concatenated_dataframe = pd.concat(labeled_dataframes_list,  ignore_index=True)
    if not Path(path.parent / 'Processed').exists():
        Path(path.parent / 'Processed').mkdir()

    concatenated_dataframe.to_csv(path.parent / 'Processed' / 'concatenated_data.csv', index=False)

if __name__ == '__main__': 
    dataframes = read_csvs('./API_ML/API/app/model/data/cube_data')
    print(len(dataframes[1]))
    process_data(dataframes)
