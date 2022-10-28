import pandas as pd
import numpy as np

submissions_path = './/submissions.csv'
ratings_path = './/ratings.csv'
df = pd.read_csv(submissions_path)


# Remove irrelevant columns from the dataset
df.drop(['contestId', 'index', 'COMPILATION_ERROR', 'SKIPPED', 'TESTING','PRESENTATION_ERROR','FAILED','CRASHED','CHALLENGED','REJECTED','PARTIAL','SECURITY_VIOLATED','INPUT_PREPARATION_CRASHED','SUBMITTED'], axis = 1, inplace=True)

users = np.array(df['username'].unique())

errors = ['WRONG_ANSWER', 'TIME_LIMIT_EXCEEDED','RUNTIME_ERROR', 'IDLENESS_LIMIT_EXCEEDED', 'MEMORY_LIMIT_EXCEEDED']
cnt = 0
for user in users:
    db = df[df['username'] == user]
    user_rating = max(800, db['user_rating'])
    
    sum_ = sum([sum(db.loc[:, i]) for i in errors])
    l_ = [(sum_ + 1) / (sum(db.loc[:, i]) + 1) for i in errors]
    # print(sum_)
    for i in db.index:
        problem_rating = db['problem_rating']
        l = [db.at[i, j] for j in errors]
        rat = problem_rating / user_rating
        s = sum(l) + db.at[i, 'OK']
        rat *= sum([l[i] * l_[i] for i in range(len(l))])
        rat /= (s * sum(l_))
        # print(i)
        cnt += 1
        rat = min(rat, 1)
        rat *= 1000
        rat = max(rat, 100)
        rat /= 20
        df.at[i, 'rating'] = np.int8(round(rat))
    if (cnt % 100000 == 0):
        print(cnt)
    # print(f"User {user} has been processed.")
    
# Removing the used columns
df.drop(['OK', 'WRONG_ANSWER', 'TIME_LIMIT_EXCEEDED', 'RUNTIME_ERROR', 'MEMORY_LIMIT_EXCEEDED', 'IDLENESS_LIMIT_EXCEEDED'], axis=1, inplace=True)

# Saving the dataframe to csv
df.to_csv(ratings_path, index=False)