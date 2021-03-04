import pandas as pd
import pickle
import os

def load_dataset():
    if 'pickeledDataset' in os.listdir('.'):
        return pickle.load(open('pickeledDataset', 'rb'))
    
    filelist = os.listdir('dataset') 
    df_list = [pd.read_json('dataset/'+file, lines=True) for file in filelist]
    df = pd.concat(df_list).reset_index(drop=True, inplace=False)

    fix_all_none(df, ['category', 'documentId', 'publishtime'])
    df['activeTime'].fillna(value=df['activeTime'].mean(), inplace=True)

    pickle.dump(df, open('pickeledDataset', 'wb'))
    return df

def fix_none(df, col_name):
    d = df.loc[(df[col_name].isna()) | (df[col_name].isnull()), ['url', col_name]].drop_duplicates()
    d = pd.DataFrame(d)
    urls = list(d['url'])
    
    elements = pd.DataFrame(df.loc[df['url'].isin(urls) & (df[col_name].isna() == False), ['url', col_name]].drop_duplicates())
    
    for index, row in elements.iterrows():
        df.loc[df['url'] == row['url'], col_name] = row[col_name]

def fix_all_none(col_names):
    for col_name in col_names:
            fix_none(col_name)


print(load_dataset())