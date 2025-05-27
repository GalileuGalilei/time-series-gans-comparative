#load output.csv and split into train and test csvs and save them into cyberdata folder
import os
import pandas as pd

# Load the data
data = pd.read_csv('output.csv', low_memory=False)

split_index = int(0.7 * len(data))  # 70% for training, 30% for testing
train = data[:split_index]
test = data[split_index:]

# Save the data
train.to_csv('train.csv', index=False)
test.to_csv('test.csv', index=False)

