class NaiveBayes:
    def __init__(self, alpha = 1):
        self.alpha = 1
        self.bow_spam = {}
        self.bow_non_spam = {}
        self.spam_prob = None
        self.non_spam_prob = None
        self.total_spam = None
        self.total_non_spam = None
    def fit(self, x, y):
        x = x.reset_index(drop=True)
        y = y.reset_index(drop=True)
        x = x.apply(lambda x: x.lower().replace('.','').replace(',', ''))
        #consider y = 1 as spam
        spam = 0
        non_spam = 0
        for i in range(len(y)):
            if y[i] == 1:
                for word in x[i].split():
                    if word in self.bow_spam:
                        spam += 1
                        self.bow_spam[word] += 1
                    else:
                        spam += self.alpha + 1
                        self.bow_spam[word] = 1 + self.alpha
            else:
                for word in x[i].split():
                    if word in self.bow_non_spam:
                        non_spam += 1
                        self.bow_non_spam[word] += 1
                    else:
                        non_spam += self.alpha + 1
                        self.bow_non_spam[word] = 1 + self.alpha
        self.total_spam = spam
        self.total_non_spam = non_spam
        self.spam_prob = (y == 1).sum() / len(y)
        self.non_spam_prob = (y == 0).sum() / len(y)

    def predict(self, x):
        x = x.reset_index(drop=True)
        x = x.apply(lambda x: x.lower().replace('.','').replace(',', ''))
        predictions = []
        for i in range(len(x)):
            bow = x[i].split()
            #compute it being spam
            spam = self.spam_prob
            for word in bow:
                if word in self.bow_spam:
                    spam *= (self.bow_spam[word] / self.total_spam)
                else:
                    spam *= (self.alpha / self.total_spam)
            non_spam = self.non_spam_prob
            #compute it being non spam
            for word in bow:
                if word in self.bow_non_spam:
                    non_spam *= (self.bow_non_spam[word] / self.total_non_spam)
                else:
                    non_spam *= (self.alpha / self.total_non_spam)
            if (spam <= non_spam):
                predictions.append(0)
            else:
                predictions.append(1)
        return predictions
