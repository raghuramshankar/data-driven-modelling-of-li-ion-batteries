from numpy import NaN
import pandas as pd

df = pd.read_csv('datasets/lg-18650hg2/LG_HG2_Original_Dataset_McMasterUniversity_Jan_2020/25degC/549_C20DisCh.csv', skiprows=28)
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df = df.drop(0)

print(df['Step'])
print('Done')