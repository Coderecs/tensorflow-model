from ProblemLens import ProblemLens
from RBMAlgorithm import RBMAlgorithm
from surprise import NormalPredictor
from Evaluator import Evaluator
from EvaluationData import EvaluationData
import random
import numpy as np

def de_hash(hash_val):
    contest = hash_val // 80
    offset = hash_val % 80
    index = chr(offset // 3 + 65)
    if offset % 3 == 1:
        index += '1'
    elif offset % 3 == 2:
        index += '2'
    return [contest, index]

def LoadProblemLensData():
    ml = ProblemLens()
    print("Loading problem ratings...")
    data = ml.loadProblemLensLatestSmall()
    print("\nComputing problem popularity ranks so we can measure novelty later...")
    rankings = ml.getPopularityRanks()
    return (ml, data, rankings)

class model(object):


	def train_model(self):

		np.random.seed(0)
		random.seed(0)

		# Load up common data set for the recommender algorithms
		(ml, evaluationData, rankings) = LoadProblemLensData()

		# Construct an Evaluator to, you know, evaluate them
		self.dataset = EvaluationData(evaluationData, rankings)
		#RBM
		print("\nUsing recommender RBM")
		self.RBM = RBMAlgorithm(epochs=20)
		trainset = self.dataset.GetFullTrainSet()
		            
		print("\nBuilding recommendation model...")

		self.RBM.fit(trainset)

	def recs(self, testSubject, k):
		print("Computing recommendations...")
		testSet = self.dataset.GetAntiTestSetForUser(testSubject)

		# predictions = self.RBM.test(testSet)
		predictions = [(i, self.RBM.estimate(u, i)) for (u, i, __) in testSet]

		recommendations = []

		print ("\nWe recommend:")
		for problemID, estimatedRating in predictions:
			intProblemID = int(problemID)
			recommendations.append((intProblemID, estimatedRating))


		recommendations.sort(key=lambda x: x[1], reverse=True)

		li = []
		for ratings in recommendations[:k]:
			li.append(de_hash(ratings[0]))
			print(de_hash(ratings[0]))

		return li
# Fight!
# Only to evaluate perfomance
# evaluator.Evaluate(True) 

# This will print our top recommended problems.
# print("Recommendations for harasees_singh are :-")
# evaluator.SampleTopNRecs(ml, k= k)
# testSubject = 'um_op'
# print("Recommendations for um_op are :-")
# evaluator.SampleTopNRecs(ml, testSubject, k)
if __name__ == '__main__':
	final = model()
	final.train_model()
	k = 20
	users = ['harasees_singh', 'ishwarendra', 'naresh6436']
	for i in users:
		print(f'Recommendations for user {i} are :-')
		final.recs(i, k)
