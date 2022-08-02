from ProblemLens import ProblemLens
from RBMAlgorithm import RBMAlgorithm
from surprise import NormalPredictor
from Evaluator import Evaluator

import random
import numpy as np

def LoadProblemLensData():
    ml = ProblemLens()
    print("Loading problem ratings...")
    data = ml.loadProblemLensLatestSmall()
    print("\nComputing problem popularity ranks so we can measure novelty later...")
    rankings = ml.getPopularityRanks()
    return (ml, data, rankings)

np.random.seed(0)
random.seed(0)

# Load up common data set for the recommender algorithms
(ml, evaluationData, rankings) = LoadProblemLensData()
testSubject = 'um_op'
k = 20

# Construct an Evaluator to, you know, evaluate them
evaluator = Evaluator(evaluationData, rankings)

#RBM
RBM = RBMAlgorithm(epochs=20)
evaluator.AddAlgorithm(RBM, "RBM")

# Fight!
# Only to evaluate perfomance
# evaluator.Evaluate(True) 

# This will print our top recommended problems.
evaluator.SampleTopNRecs(ml, testSubject, k)
