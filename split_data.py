#load output.csv and split into train and test csvs and save them into cyberdata folder
import os
import pandas as pd

# Load the data
data = pd.read_csv('cyberdata/output.csv')

# Split the data
cutoff = int(0.8 * len(data))
train = data.iloc[:cutoff]
test = data.iloc[cutoff:]

# Save the data
train.to_csv('cyberdata/train.csv', index=False)
test.to_csv('cyberdata/test.csv', index=False)

