import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

def calculate_entropy(data, time_scale): 
    #falta mandar el rango de empiezar y terminar el ciclo 
    multiplier = 1/time_scale
    addition_of_x = 0

    for x_data in data:
        addition_of_x += x_data

    entropy = multiplier * addition_of_x
    
    return entropy #se retorna un escalar

def calculate_multiscale_entropy(channel, scale_factor, num_samples): 
    entropies_multiscale = []

    #Calculamos entropias de diferentes escalas dividiendo los datos en segmentos y calculando la entropia de cada segmento 
    for i in range(0, num_samples, scale_factor):
        entropies_multiscale.append(calculate_entropy(channel.iloc[i:i+scale_factor], scale_factor))

    # Drop the last element of the list
    entropies_multiscale.pop()

    return entropies_multiscale

def get_channels(data):
    scale_factor = 256
    num_samples, num_channels = data.shape
    entropies_multiscale = {}
    #recorremos los canales para calcular sus entropias 
    for i in range(0, num_channels):
        # Get the i column
        column = data.iloc[:, i]
        entropies_multiscale[column.name] = calculate_multiscale_entropy(channel=column, scale_factor=scale_factor, num_samples=num_samples)

    # Add the time column based on the scale factor / number of samples
    entropies_multiscale['Time'] = [i/scale_factor + 1 for i in range(0, num_samples, scale_factor)]
    entropies_multiscale['Time'].pop()

    return pd.DataFrame(entropies_multiscale)


# Function to read all csv files and return a list with file names and dataframes
def read_csv_files(path:str):
    # Get all csv files in the path
    csv_files = Path(path).glob('*.csv')
    # Create a list of dataframes
    dfs = [pd.read_csv(file) for file in csv_files]
    # Get the name of each file inside the path
    p = Path(path)
    csv_files = list(p.glob('*.csv'))
    return [dfs, csv_files]

def process_dataframe(df, label:str):
    # Rename the Channel x column to Chxlabel
    rename = {f'Channel {i}': f'CH{i}{label}' for i in range(1, 15)}
    # Rename the columns
    for key, value in rename.items():
        df[value] = df[key]
        del df[key]
    return df

def process_row(row, iteration:int, label:str):
    # Delete the Time column
    del row['Time']
    # Rename the Channel x column to Chxlabel
    rename = {f'Channel {i}': f'CH{i}{label}' for i in range(1, 15)}
    # Add the iteration number as Time column
    row['Time'] = iteration + 1

    for key, value in rename.items():
        row[value] = row[key]
        del row[key]

    return row

def place_classification_column(merge_df): 

    # Create classification labels for each row 
    column = ['open_hand' for _ in range(20)] + ['rest' for _ in range(20)] + ['close_hand' for _ in range(20)] + ['rest' for _ in range(10)]
    
    # Set the classification column values 
    merge_df['Classification'] = column
    return merge_df

# Function to delete rename and apply changes to the dataframes
def process_dataframes(dataframes):
    drop_columns = ['Epoch', 'Event Id', 'Event Date', 'Event Duration']
    new_merge_df = pd.DataFrame()
    for i in range(len(dataframes[0])):
        means = []
        stds = []
        # Copy the dataframe 
        df = dataframes[0][i].copy()
        # Delete the columns that are not needed
        df.drop(columns=drop_columns, inplace=True)
        # Rename the Time:256Hz column to Time
        df.rename(columns={'Time:512Hz':'Time'}, inplace=True)
        # Calculate the iterations dividing the number of rows in the dataframe by 256
        iterations = int(len(df) / 256)
        # Iterate each 256 rows and calculate the mean of each column
        for j in range(iterations):
            # Calculate the start and end of the 256 rows
            start = j * 256
            end = start + 256
            # Calculate the mean of each column
            means.append(process_row(df.iloc[start:end].mean(), j, 'MEAN'))
            # Calculate the standard deviation of each column
            stds.append(process_row(df.iloc[start:end].std(), j, 'STD'))
        mean_dataframe = pd.DataFrame(means)

        std_dataframe = pd.DataFrame(stds)
        # Concatenate the mean and std dataframes
        new_dataframe = pd.merge(mean_dataframe, std_dataframe, on='Time')

        # Get the entropy multiscale of each channel
        df1 = df.copy()
        df1.drop(columns=['Time'], inplace=True)
        multiescale =  process_dataframe(get_channels(pd.DataFrame(scaler.fit_transform(df1), columns=df1.columns)), 'ENTROPY')

        # Concatenate the entropy multiscale dataframe to the new dataframe
        new_dataframe = pd.merge(new_dataframe, multiescale, on='Time')

        # Ignore first seconds of resting state
        new_dataframe = new_dataframe.iloc[10:]
        
        # Add classification column
        new_dataframe = place_classification_column(new_dataframe)

        # Remove all rows with rest in the classification column
        new_dataframe = new_dataframe[new_dataframe.Classification != 'rest']

        # Save the new dataframe to a csv file
        path = dataframes[1][i]
        # Check if exist the folder processed in the parent folder of the csv file
        if not Path(path.parent / 'Processed').exists():
            # Create the folder processed
            Path(path.parent / 'Processed').mkdir()
        # Save the new dataframe to a csv file
        new_dataframe.to_csv(path.parent / 'Processed' / path.name, index=False)

        # Concatenate the new dataframe to the merge dataframe ignoring the Time column and index
        new_dataframe.drop(columns=['Time'], inplace=True)
        new_merge_df = pd.concat([new_merge_df, new_dataframe], ignore_index=True)
    
    # Change the open_hand and close_hand values to 0 and 1
    new_merge_df.Classification.replace({'open_hand':0, 'close_hand':1}, inplace=True)

    # Shuffle the dataframe # DELETE THIS COMPLETLY 
    new_merge_df = new_merge_df.sample(frac=1).reset_index(drop=True)

    # Save the merge dataframe to a csv file
    new_merge_df.to_csv(path.parent / 'Processed' / 'EEG.csv', index=False)

    # Save the 20% of the dataframe to a csv file
    new_merge_df.sample(frac=0.2).to_csv(path.parent / 'Processed' / 'EEG_test.csv', index=False, header=False)

    # Save the 80% of the dataframe to a csv file
    new_merge_df.sample(frac=0.8).to_csv(path.parent / 'Processed' / 'EEG_train.csv', index=False, header=False)

if __name__ == '__main__':
    # Read all csv files
    dataframes = read_csv_files('data/Records/Raw')
    # Process the dataframes
    process_dataframes(dataframes)