import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--data_name', type=str, default='agnews')

args = parser.parse_args()

# Read data
train_df = pd.read_csv(f'data/{args.data_name}/train.csv')
test_df = pd.read_csv(f'data/{args.data_name}/test.csv')

for data_type, df in zip(['train', 'test'], [train_df, test_df]):
    print(f"============== {data_type} ==============")
    print(f"The columns of the data: {df.columns}")
    for column_name in ["Title", "Description"]:
        df["length"] = df[column_name].apply(lambda x: len(x.split()))
        print(f'Avg. Length of data in column `{column_name}`: {df["length"].mean():.2f}')