from DecisionTrees import DecisionTrees
import numpy as np
from collections import Counter

class RandomForest:
    def __init__(self, n_trees):
        self.n_trees = n_trees
        self.trees = []
    def fit(self, x, y):
        for i in range(self.n_trees):
            tree = DecisionTrees()
            x_sample, y_sample = self.make_sample(x,y)
            tree.fit(x_sample, y_sample)
            self.trees.append(tree)
    def make_sample(self, x, y):
        n_samples = x.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return x.iloc[idxs], y.iloc[idxs]
    def most_common_label(self, y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common
    def predict(self,x):
        predictions = np.array([tree.predict(x) for tree in self.trees])
        tree_preds = np.swapaxes(predictions, 0, 1)
        predictions = np.array([self.most_common_label(pred) for pred in tree_preds])
        return predictions