import os
import csv
import sys
import re

from surprise import Dataset
from surprise import Reader

from collections import defaultdict

class ProblemLens:

    ratingsPath = '../ml-latest-small/ratings.csv'
    problemsPath = '../ml-latest-small/problems.csv'
    
    def loadProblemLensLatestSmall(self):
        
        # Loads the dataset for the training purposes

        # Look for files relative to the directory we are running from
        os.chdir(os.path.dirname(sys.argv[0]))

        ratingsDataset = 0

        reader = Reader(line_format='user problem rating timestamp', sep=',', skip_lines=1)

        ratingsDataset = Dataset.load_from_file(self.ratingsPath, reader=reader)

        return ratingsDataset

    def getUserRatings(self, user):
        
        # Returns the ratings(the one generated during preprocessing) corresponding to each problem attempted by him/her.
        userRatings = []
        hitUser = False
        with open(self.ratingsPath, newline='') as csvfile:
            ratingReader = csv.reader(csvfile)
            next(ratingReader)
            for row in ratingReader:
                userID = int(row[0])
                if (user == userID):
                    problemID = int(row[1])
                    rating = float(row[2])
                    userRatings.append((problemID, rating))
                    hitUser = True
                if (hitUser and (user != userID)):
                    break

        return userRatings

    def getPopularityRanks(self):
        # Returns the popularity rankings of the problems
        ratings = defaultdict(int)
        rankings = defaultdict(int)
        with open(self.ratingsPath, newline='') as csvfile:
            ratingReader = csv.reader(csvfile)
            next(ratingReader)
            for row in ratingReader:
                problemID = int(row[1])
                ratings[problemID] += 1
        rank = 1
        for problemID, ratingCount in sorted(ratings.items(), key=lambda x: x[1], reverse=True):
            rankings[problemID] = rank
            rank += 1
        return rankings
    
    def getTags(self):
        # Basically performs one hot encoding for the tags of the problems.
        tags = defaultdict(list)
        tagIDs = {}
        maxTagID = 0
        with open(self.problemsPath, newline='', encoding='ISO-8859-1') as csvfile:
            problemReader = csv.reader(csvfile)
            next(problemReader)  #Skip header line
            for row in problemReader:
                problemID = int(row[0])
                tagList = row[2].split('|')
                tagIDList = []
                for tag in tagList:
                    if tag in tagIDs:
                        tagID = tagIDs[tag]
                    else:
                        tagID = maxTagID
                        tagIDs[tag] = tagID
                        maxTagID += 1
                    tagIDList.append(tagID)
                tags[problemID] = tagIDList
        # Convert integer-encoded tag lists to bitfields that we can treat as vectors.
        for (problemID, tagIDList) in tags.items():
            bitfield = [0] * maxTagID
            for tagID in tagIDList:
                bitfield[tagID] = 1
            tags[problemID] = bitfield            
        
        return tags
    
    def getRatings(self):
        # Returns the codeforces(or equivalent) rating of the problem.
        p = re.compile(r"(?:\((\d{4})\))?\s*$")
        ratings = defaultdict(int)
        with open(self.problemsPath, newline='', encoding='ISO-8859-1') as csvfile:
            problemReader = csv.reader(csvfile)
            next(problemReader)
            for row in problemReader:
                problemID = int(row[0])
                title = row[1]
                m = p.search(title)
                rating = m.group(1)
                if rating:
                    ratings[problemID] = int(rating)
        return ratings
    
