from matplotlib import pyplot as plt
import pandas as pd
from pathlib import Path

p = Path('data/Records/Raw/Processed')
csv_files = list(p.glob('Gerardo*.csv'))

columns = ["CH1ENTROPY","CH2ENTROPY","CH3ENTROPY","CH4ENTROPY","CH5ENTROPY","CH6ENTROPY","CH7ENTROPY","CH8ENTROPY","CH9ENTROPY","CH10ENTROPY","CH11ENTROPY","CH12ENTROPY","CH13ENTROPY","CH14ENTROPY"]

for file in csv_files:
    df = pd.read_csv(file, usecols=columns)

    # Create a new column with the time in seconds from 1 to 40
    df['Time'] = [i for i in range(1, 41)]

    # Plot all the means columns
    plt.figure(figsize=(10, 5))
    plt.plot(df.Time, df.CH1ENTROPY, label='CH1ENTROPY', marker='o')
    plt.plot(df.Time, df.CH2ENTROPY, label='CH2ENTROPY', marker='o')
    plt.plot(df.Time, df.CH3ENTROPY, label='CH3ENTROPY', marker='o')
    plt.plot(df.Time, df.CH4ENTROPY, label='CH4ENTROPY', marker='o')
    plt.plot(df.Time, df.CH5ENTROPY, label='CH5ENTROPY', marker='o')
    plt.plot(df.Time, df.CH6ENTROPY, label='CH6ENTROPY', marker='o')
    plt.plot(df.Time, df.CH7ENTROPY, label='CH7ENTROPY', marker='o')
    plt.plot(df.Time, df.CH8ENTROPY, label='CH8ENTROPY', marker='o')
    plt.plot(df.Time, df.CH9ENTROPY, label='CH9ENTROPY', marker='o')
    plt.plot(df.Time, df.CH10ENTROPY, label='CH10ENTROPY', marker='o')
    plt.plot(df.Time, df.CH11ENTROPY, label='CH11ENTROPY', marker='o')
    plt.plot(df.Time, df.CH12ENTROPY, label='CH12ENTROPY', marker='o')
    plt.plot(df.Time, df.CH13ENTROPY, label='CH13ENTROPY', marker='o')
    plt.plot(df.Time, df.CH14ENTROPY, label='CH14ENTROPY', marker='o')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Entropy')
    plt.title('Entropy of each channel')

    plt.show()
input()