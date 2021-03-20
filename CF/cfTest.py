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
        self.scaler = StandardScaler()
        
        self.users['activeTime'] = self.scaler.fit_transform(np.array(self.users['activeTime']).reshape(-1, 1))

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
        testset = pd.DataFrame(self.testset, columns=['uid', 'iid', 'rating'])
        return self.algo.test(testset.values.tolist())
        
    def predict_user(self, user):
        testset = pd.DataFrame(self.testset, columns=['uid', 'iid', 'rating'])
        return self.algo.test(testset[testset['uid'] == user].values.tolist())

    def predictions_to_dataframe(self, predictions):
        return pd.DataFrame(predictions)

    def sort_predictions(self, predictions):
        return predictions.sort_values("est", ascending=False, inplace=False)

    def scale_predictions(self, predictions):
        predictions['r_ui'] = self.scaler.inverse_transform(np.array(predictions['r_ui']).reshape(-1,1))
        predictions['est'] = self.scaler.inverse_transform(np.array(predictions['est']).reshape(-1,1))
        return predictions

    def get_top_n(self, predictions, user=None, n=10):
        if user is None:
            return predictions.sort_values('est', inplace=True, ascending=False).head(n)
        return predictions[predictions['uid'] == user].sort_values('est', inplace=True, ascending=False).head(n)

    def precision_recall(self, predictions, k=10, threshold=44):
        user_est_true = defaultdict(list)
        for i, row in predictions.iterrows():
            user_est_true[row['uid']].append((row['est'], row['r_ui']))
        
        precisions = dict()
        recalls = dict()
        for uid, user_ratings in user_est_true.items():
            user_ratings.sort(key=lambda x: x[0], reverse=True)

            n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

            n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

            n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                                        for (est, true_r) in user_ratings[:k])

            precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0

            recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

        return precisions, recalls

    def ARHR(self, predictions, k=15, threshold=44):
        user_est_true = defaultdict(list)
        for i, row in predictions.iterrows():
            user_est_true[row['uid']].append((row['est'], row['r_ui']))

        reciprocal_rank = 0
        for uid in user_est_true:
            for i, user_ratings in enumerate(user_est_true[uid][:k]):
                # print((user_ratings[0] >= threshold and user_ratings[1] >= threshold)/(i+1))
                reciprocal_rank += (user_ratings[0] >= threshold and user_ratings[1] >= threshold)/(i+1)

        return reciprocal_rank/len(list(user_est_true.keys()))
                    

if __name__ == '__main__':
    cf = CF(15, 5, 'pearson', True)
    cf.train()
    testset = pd.DataFrame(cf.testset, columns=['uid', 'iid', 'rating'])
    user1test = testset[testset['uid'] == 'cx:13576697471061598567701:1msq47q99r2b6']
    user1id = 'cx:13576697471061598567701:1msq47q99r2b6'
    user1anti = testset[testset['uid'] == user1id]
    predictions = cf.predict_all()
    accuracy.mse(predictions)
    predictions = cf.sort_predictions(cf.scale_predictions(cf.predictions_to_dataframe(predictions)))
    # print(cf.precision_recall(predictions, k=9999))
    print(cf.ARHR(predictions))
