import os
import pandas as pd
import argparse

'''
Load all the dapt2020 csv files from the data_path directory
and concatenate them into a single DataFrame.
The final DataFrame is saved to data/dapt2020.csv.
'''
def main(data_path):

    dapt2020 = pd.DataFrame()

    # Check if the data_path exists
    if not os.path.exists(data_path):
        print(f"Data path {data_path} does not exist.")
        return

    #load each csv file in data_path
    columns = []
    for file in os.listdir(data_path):
        if file.endswith('.csv'):
            file_path = os.path.join(data_path, file)
            print(f"Loading {file_path}")
            df = pd.read_csv(file_path, header=0)

            # One of the csv files from Kaggle does not have header column for some reason, so we fix this here
            if df.columns.__contains__("Timestamp"):
                columns = df.columns.to_list()
            else:
                df.columns = columns

            # Check if the DataFrame is empty
            if df.empty:
                print(f"DataFrame from {file_path} is empty. Skipping.")
                continue

            #appends the data to final_data
            dapt2020 = pd.concat([dapt2020, df], axis=0, ignore_index=True)
    # Check if final_data is empty
    if dapt2020.empty:
        print("Final DataFrame is empty after loading all CSV files.")
        return
    
    #Sort by Timestamp if it exists
    dapt2020['Timestamp'] = pd.to_datetime(dapt2020['Timestamp'], errors='coerce')
    dapt2020.sort_values(by='Timestamp', inplace=True)
    
    # save the data
    output_path = "data/dapt2020.csv"
    dapt2020.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Load DAPT2020 CSV files and save to a single DataFrame.')
    parser.add_argument('--data_path', type=str, default='data/csv', help='Path to the directory containing DAPT2020 CSV files.')
    args = parser.parse_args()
    main(args.data_path)
