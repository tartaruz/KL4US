import pandas as pd
import pickle
import os

class Data:
    def __init__(self):
        pass


    def load_dataset(self, datasetName):
        if datasetName in os.listdir('.'):
            return pickle.load(open(datasetName, 'rb'))
        
        filelist = os.listdir('dataset') 
        df_list = [pd.read_json('dataset/'+file, lines=True) for file in filelist]
        df = pd.concat(df_list).reset_index(drop=True, inplace=False)

        df['activeTime'].fillna(value=df['activeTime'].mean(), inplace=True)
        df = df[~(df['url'] == 'http://adressa.no')]
        df.drop_duplicates(subset=['userId', 'documentId'], inplace=True)
        df.reset_index(inplace=True, drop=True)
        self.fix_all_none(df, ['category', 'documentId', 'publishtime'])

        pickle.dump(df, open('pickeledDataset', 'wb'))
        return df

    def fix_none(self, df, col_name):
        d = df.loc[(df[col_name].isna()) | (df[col_name].isnull()), ['url', col_name]].drop_duplicates()
        d = pd.DataFrame(d)
        urls = list(d['url'])
        
        elements = pd.DataFrame(df.loc[df['url'].isin(urls) & (df[col_name].isna() == False), ['url', col_name]].drop_duplicates())
        
        for index, row in elements.iterrows():
            print(f'Index: {index}, col: {col_name}')
            df.loc[df['url'] == row['url'], col_name] = row[col_name]

    def fix_all_none(self, df, col_names):
        for col_name in col_names:
                self.fix_none(df, col_name)

if __name__ == '__main__':
    dataset = Data().load_dataset('pickeledDataset')
    print(dataset)