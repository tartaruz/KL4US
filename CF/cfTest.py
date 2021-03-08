from dataset import Data
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split, cross_validate
from surprise import KNNWithMeans
from surprise import accuracy
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

class CF:
    def __init__(self, k, min_k, sim, user_based):
        self.dataset = Data().load_dataset('pickeledDataset')
        self.users = self.dataset[['userId', 'documentId', 'activeTime']]
        scaler = StandardScaler()
        
        self.users['activeTime'] = scaler.fit_transform(np.array(self.users['activeTime']).reshape(-1, 1))

        self.users.columns = ['userId', 'documentId', 'rating']

        rating_scale=(self.users['rating'].min(), self.users['rating'].max())
        self.data = Dataset.load_from_df(self.users[['userId', 'documentId', 'rating']], reader=Reader(rating_scale=rating_scale))
        
        self.trainset, self.testset = train_test_split(self.data, test_size=0.2)
        self.trainsetfull = self.data.build_full_trainset()

        self.trainset_iids = list(self.trainset.all_items())
        self.trainset_uids = list(self.trainset.all_users())

        self.iid_converter = lambda x: self.trainset.to_raw_iid(x)
        self.uid_converter = lambda x: self.trainset.to_raw_uid(x)

        self.trainset_raw_iids = list(map(self.iid_converter, self.trainset_iids))
        self.trainset_raw_uids = list(map(self.uid_converter, self.trainset_uids))

        self.k = k
        self.min_k = min_k

        self.sim_option = {
            'name':sim, 'user_based':user_based
        }
        
        self.algo = KNNWithMeans(
            k = self.k, min_k = self.min_k, sim_option = self.sim_option
        )
    

    def train(self):
        self.algo.fit(self.trainset)
    

    def test(self):
        return self.algo.test(self.testset)

    def cross_validate(self):
        return cross_validate(
            algo = self.algo, data = self.data, measures=['RMSE'], 
            cv=5, return_train_measures=True
        )

    def fit(self):
        self.algo.fit(self.trainsetfull)

    def predict_all(self):
        testset = self.trainsetfull.build_anti_testset()
        predictions = self.algo.test(testset)

        return predictions

    def get_top_n(self, predictions, n=10):
        top_n = defaultdict(list)
        for uid, iid, true_r, est, _ in predictions:
            top_n[uid].append((iid, est))

        for uid, user_ratings in top_n.items():
            user_ratings.sort(key=lambda x: x[1], reverse=True)
            top_n[uid] = user_ratings[:n]

        return top_n

if __name__ == '__main__':
    cf = CF(15, 5, 'pearson', True)
    cf.train()
    accuracy.mse(cf.test())
    # testset = pd.DataFrame(cf.testset, columns=['uid', 'iid', 'rating'])
    # antitestset = pd.DataFrame(cf.trainsetfull.build_anti_testset(), columns=['uid', 'iid', 'rating'])
    # user1test = testset[testset['uid'] == 'cx:13576697471061598567701:1msq47q99r2b6']
    # user1anti = antitestset[antitestset['uid'] == 'cx:13576697471061598567701:1msq47q99r2b6']
    # user1id = 'cx:13576697471061598567701:1msq47q99r2b6'
    # antitestset = cf.trainsetfull.build_anti_testset()
    # user1anti = list(filter(lambda x: x[0] == user1id, antitestset))
    # predictions = cf.algo.test(user1anti)
    # print(cf.get_top_n(predictions))
    

    # print(testset.groupby(['uid', 'iid']).head())
    # cf.fit()
    # # TODO: This is not working.
    # for uid, user_ratings in cf.get_top_n(cf.predict_all()):
    #     print(uid, [iid for (iid, _) in user_ratings])