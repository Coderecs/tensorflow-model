import pandas as pd
import numpy as np

submissions_path = './/submissions.csv'

df = pd.read_csv(submissions_path)


# Remove irrelevant columns from the dataset
df.drop(['contestId', 'index', 'COMPILATION_ERROR', 'SKIPPED', 'TESTING','PRESENTATION_ERROR','FAILED','CRASHED','CHALLENGED','REJECTED','PARTIAL','SECURITY_VIOLATED','INPUT_PREPARATION_CRASHED','SUBMITTED'], axis = 1, inplace=True)

'''
The processing is yet to be written
'''


# Removing the used columns
df.drop(['OK', 'WRONG_ANSWER', 'TIME_LIMIT_EXCEEDED', 'RUNTIME_ERROR', 'MEMORY_LIMIT_EXCEEDED', 'IDLENESS_LIMIT_EXCEEDED'], axis=1, inplace=True)

# Saving the dataframe to csv
df.to_csv('ratings.csv', index=False)