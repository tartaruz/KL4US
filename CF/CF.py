from dataset import Data
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import KNNWithMeans
from surprise import accuracy
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

class CF:
    def __init__(self, k, min_k, sim, user_based):
        self.dataset = Data().load_dataset('pickeledDataset') # Load pickeled dataset (or preprocess the dataset if the pickle does not exist)
        self.users = self.dataset[['userId', 'documentId', 'activeTime']] # Create a user dataframe
        
        self.scaler = StandardScaler() # Init scaler
        
        self.users['activeTime'] = self.scaler.fit_transform(np.array(self.users['activeTime']).reshape(-1, 1)) # Scale all active times 

        self.users.columns = ['userId', 'documentId', 'rating'] # Change "activeTime" to "rating"

        rating_scale=(self.users['rating'].min(), self.users['rating'].max()) # Rating scale to be used (min(activeTime) - max(activeTime))
        self.data = Dataset.load_from_df(self.users[['userId', 'documentId', 'rating']], reader=Reader(rating_scale=rating_scale)) # Set CF data as user dataframe
        
        self.trainset, self.testset = train_test_split(self.data, test_size=0.2) # Split data into 80% train and 20% test
        self.trainsetfull = self.data.build_full_trainset() # Build full trainset

        self.trainset_iids = list(self.trainset.all_items()) # Scikit-learn uses "inner ids", get all the inner ids for items
        self.trainset_uids = list(self.trainset.all_users()) # Get all inner ids for users 

        self.iid_converter = lambda x: self.trainset.to_raw_iid(x) # Converter for innerid and actual id for items
        self.uid_converter = lambda x: self.trainset.to_raw_uid(x) # Converter for innerid and actual id for users

        self.trainset_raw_iids = list(map(self.iid_converter, self.trainset_iids)) # Store list with raw item ids
        self.trainset_raw_uids = list(map(self.uid_converter, self.trainset_uids)) # Store list with raw user ids

        self.k = k # Set k to be used in KNN
        self.min_k = min_k # Minimum neighbours to consider.

        # Define similarity options
        self.sim_option = {
            'name':sim, 'user_based':user_based
        }
        
        # Define algorithm to be used in the CF.
        self.algo = KNNWithMeans(
            k = self.k, min_k = self.min_k, sim_option = self.sim_option
        )
    

    def train(self):
        """Train model on trainset"""

        self.algo.fit(self.trainset)
    

    def test(self):
        """Predict items"""

        return self.algo.test(self.testset)

    # def fit(self):
    #     """Train """
    #     self.algo.fit(self.trainsetfull)

    def predict_all(self):
        """Recommend for all users"""

        testset = pd.DataFrame(self.testset, columns=['uid', 'iid', 'rating'])
        return self.algo.test(testset.values.tolist())
        
    def predict_user(self, user):
        """Recommend for one user"""

        testset = pd.DataFrame(self.testset, columns=['uid', 'iid', 'rating'])
        return self.algo.test(testset[testset['uid'] == user].values.tolist())

    def predictions_to_dataframe(self, predictions):
        """Map predictions to Pandas DataFrame"""

        return pd.DataFrame(predictions)

    def sort_predictions(self, predictions):
        """Sort the predictions"""

        return predictions.sort_values("est", ascending=False, inplace=False)

    def scale_predictions(self, predictions):
        """Scale predictions back to original aciveTime"""

        predictions['r_ui'] = self.scaler.inverse_transform(np.array(predictions['r_ui']).reshape(-1,1))
        predictions['est'] = self.scaler.inverse_transform(np.array(predictions['est']).reshape(-1,1))
        return predictions

    def get_top_n(self, predictions, user=None, n=10):
        """Get top N predictions"""

        if user is None:
            return predictions.sort_values('est', inplace=True, ascending=False).head(n)
        return predictions[predictions['uid'] == user].sort_values('est', inplace=True, ascending=False).head(n)

    def precision_recall(self, predictions, k=10, threshold=44):
        """Calculate precision@k and recall@k for predictions"""
        
        user_est_true = defaultdict(list)
        for _, row in predictions.iterrows():
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

    def ARHR(self, predictions, k=5, threshold=44):
        user_est_true = defaultdict(list)
        for i, row in predictions.iterrows():
            user_est_true[row['uid']].append((row['est'], row['r_ui']))

        reciprocal_rank = 0
        for uid in user_est_true:
            for i, user_ratings in enumerate(user_est_true[uid][:k]):
                reciprocal_rank += (user_ratings[0] >= threshold and user_ratings[1] >= threshold)/(i+1)

        return reciprocal_rank/len(list(user_est_true.keys()))
                    

if __name__ == '__main__':
    k = 20 # K - neighbors.
    min_k = 5 # Minimum neighbours to consider.
    similarity_measure = 'pearson' # What similarity measure that should be used in KNN
    user_based = True # User based or item-based

    cf = CF(k, min_k, similarity_measure, user_based) # Create CF-object
    cf.train() # Traing the model

    predictions = cf.predict_all() # Recommend articles for all users in dataset.

    accuracy.mse(predictions) # Print MSE

    predictions = cf.sort_predictions(cf.scale_predictions(cf.predictions_to_dataframe(predictions))) # Process predictions.
    
    precisions, recalls = cf.precision_recall(predictions, k=k) # Get precision and recall

    print(f'Precision: {sum(precisions.values())/len(list(precisions.keys()))}') # Print mean precision
    print(f'Recall: {sum(recalls.values())/len(list(recalls.keys()))}') # Print mean recall
    print(f'ARHR: {cf.ARHR(predictions, k=k)}') # Print ARHR
