from surprise import AlgoBase
from surprise import PredictionImpossible
import numpy as np
from RBM import RBM

class RBMAlgorithm(AlgoBase):

    def __init__(self, epochs=20, hiddenDim=100, learningRate=0.001, batchSize=100, sim_options={}):
        AlgoBase.__init__(self)
        self.epochs = epochs
        self.hiddenDim = hiddenDim
        self.learningRate = learningRate
        self.batchSize = batchSize
        
    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)

        self.numUsers = trainset.n_users
        self.numItems = trainset.n_items
        self.predictedRatings = np.full((self.numUsers, self.numItems), fill_value = np.float32(-1))
        self.trainingMatrix = np.zeros([self.numUsers, self.numItems, 11], dtype=np.float32)
        
        for (uid, iid, rating) in trainset.all_ratings():
            # adjustedRating = int(float(rating)*2.0) - 1
            try:
                self.trainingMatrix[int(uid), int(iid), int(rating)] = 1
            except:
                print(uid, iid, rating)
        
        # Flatten to a 2D array, with nodes for each possible rating type on each possible item, for every user.
        self.trainingMatrix = np.reshape(self.trainingMatrix, [self.trainingMatrix.shape[0], -1])
        
        # Create an RBM with (num items * rating values) visible nodes
        self.rbm = RBM(self.trainingMatrix.shape[1], hiddenDimensions=self.hiddenDim, learningRate=self.learningRate, batchSize=self.batchSize, epochs=self.epochs)
        self.rbm.Train(self.trainingMatrix)

    def predict_ratings(self, uiid):
        # for uiid in range(trainset.n_users):
        #     if (uiid % 50 == 0):
        #         print("Processing user ", uiid)
        recs = self.rbm.GetRecommendations([self.trainingMatrix[uiid]])
        recs = np.reshape(recs, [self.numItems, 11])
        
        for itemID, rec in enumerate(recs):
            # The obvious thing would be to just take the rating with the highest score:                
            #rating = rec.argmax()
            # ... but this just leads to a huge multi-way tie for 5-star predictions.
            # The paper suggests performing normalization over K values to get probabilities
            # and take the expectation as your prediction, so we'll do that instead:
            normalized = self.softmax(rec)
            rating = np.average(np.arange(11), weights=normalized)
            self.predictedRatings[uiid, itemID] = (rating)
        
        return self


    def estimate(self, u, i):
        u = self.trainset.to_inner_uid(u)
        i = self.trainset.to_inner_iid(i) 
         
        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible('User and/or item is unknown.')
        if (self.predictedRatings[u, i] == -1):
            self.predict_ratings(u)
        rating = self.predictedRatings[u, i]
        
        if (rating < 0.001):
            raise PredictionImpossible('No valid prediction exists.')
            
        return rating
    