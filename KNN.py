import math
class KNN:
    def __init__(self, k):
        self.x_train = None
        self.y_train = None
        self.k = k
    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
    def predict(self, x_test):
        ret = []
        for i, row in x_test.iterrows():
            distCollection = {}
            for j, drow in self.x_train.iterrows():
                d = 0
                for col in self.x_train.columns:
                    d += (row[col] - drow[col])**2
                dist = math.sqrt(d)
                distCollection[j] = dist
            sortedDistCollection = dict(sorted(distCollection.items(), key=lambda item: item[1]))
            group_keys = list(sortedDistCollection.keys())[:self.k]
            group = {key: sortedDistCollection[key] for key in group_keys}
            votes = {}
            for key, value in group.items():
                if self.y_train[key] in votes:
                    votes[self.y_train[key]] += 1
                else:
                    votes[self.y_train[key]] = 1
            max_vote = max(votes, key=votes.get)
            ret.append(max_vote)
        return ret