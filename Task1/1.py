import pandas as pd

df = pd.read_csv('data.csv') # read the data
print(df)


def median(list_vales):
    n_num = list_vales
    n = len(n_num)
    n_num.sort()

    if n % 2 == 0:
        median1 = n_num[n // 2]
        median2 = n_num[n // 2 - 1]
        median = (median1 + median2) / 2
    else:
        median = n_num[n // 2]
    return median

def average(list_val):
    return sum(list_val) / len(list_val)

def cleandata(df):
    nan_rows = df[df['height'].isnull()]  # get the row which is have null values

    row = df.iloc[4].tolist()  # the index of the row which is null

    if row[1] == 1:  # check if the gender is Male/female then remove the height values which are female
        new_df = df.loc[df['gender'] == 1]
        new_df = new_df.drop(nan_rows.index[0])
        height = new_df['height'].tolist()  # get all the values of height except the NAN value
        median_val = median(height)
        average_val = average(height)
        # print("the median value", median_val)
        # print(" the average value", int(average_val))
        df.at[nan_rows.index[0], 'height'] = median_val  # update the  corrupted VALUE
        df_median = df.copy()  # make new table with median value
        df.at[nan_rows.index[0], 'height'] = int(average_val)  # update the  corrupted VALUE
        df_avg = df.copy()  # make new table with avg value
        return df_median, df_avg, int(average_val), int(median_val)

    elif row[1] == 0:  # check if the gender is Male/female then remove the height values which are male
        new_df = df.loc[df['gender'] == 0]
        new_df = new_df.drop(nan_rows.index[0])
        height = new_df['height'].tolist()
        median_val = median(height)
        average_val = average(height)
        print(median_val)
        df.at[nan_rows.index[0], 'height'] = median_val # update the  corrupted VALUE
        df_median = df.copy()  # make new table with median value
        df.at[nan_rows.index[0], 'height'] = int(average_val)
        df_avg = df.copy()  # make new table with median value
        return df_median, df_avg, int(average_val), int(median_val)


df_median, df_avg, average_val, median_val = cleandata(df)
print("------------------------------------------------------------")
print("the AVERAGE value is:", average_val)
print(df_avg)
print("------------------------------------------------------------")
print("the median value is:", median_val)
print(df_median)
