import pandas as pd
import numpy as np

# output csv
# including cleaned data
def generate_csv(outputfilepath, df):
    df.to_csv(outputfilepath, sep=',', encoding='utf-8')

# df = pd.read_csv(r"C:\Users\ZHA244\Coding\QLD\baffle_creek\baffle-creek-buoy-quality-2013-all-forpca.csv")
#
# generate_csv(r"C:\Users\ZHA244\Coding\QLD\baffle_creek\baffle-creek-buoy-quality-2013-all-forpca-120min.csv",
#              df.iloc[::4])



df = pd.read_csv(r"C:\Users\ZHA244\Coding\QLD\baffle_creek\rain_season.csv")

generate_csv(r"C:\Users\ZHA244\Coding\QLD\baffle_creek\rain_season-120min.csv",
             df.iloc[::4])


