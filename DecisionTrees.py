def gini(df, column):
    x11 = 0
    x10 = 0
    x01 = 0
    x00 = 0
    for i in range(len(df)):
        if df.iloc[i][column] == 1 and df.iloc[i]['target'] == 1:
            x11 += 1
        elif df.iloc[i][column] == 1 and df.iloc[i]['target'] == 0:
            x10 += 1
        elif df.iloc[i][column] == 0 and df.iloc[i]['target'] == 1:
            x01 += 1
        elif df.iloc[i][column] == 0 and df.iloc[i]['target'] == 0:
            x00 += 1
    total = x11+x00+x01+x10
    if (total == 0):
        return 1
    g_left = 1
    if (x11 + x10 > 0):
        g_left = 1 - (x11 / (x11 + x10)) ** 2 - (x10 / (x11 + x10)) ** 2
    g_right = 1
    if (x01 + x00 < 1):
        g_right = 1 - (x01 / (x01 + x00)) ** 2 - (x00 / (x01 + x00)) ** 2
    g = g_left*((x11+x10)/total) + g_right*((x00+x01)/total)
    return g

class Node:
    def __init__(self, feature = None, left=None, right=None):
        self.feature = feature
        self.left = left
        self.right = right

class DecisionTrees:
    def __init__(self):
        self.head = None
        self.cols = []
        self.df = None
    def best_split(self, df):
        best = self.cols[0]
        best_gini = gini(df, best)
        for col in self.cols:
            g = gini(df, col)
            if (g < best_gini):
                best = col
                best_gini = g
        return best

    def fit(self, x_df, y_df):
        for col in x_df.columns:
            self.cols.append(col)
        new_df = x_df
        new_df['target'] = y_df
        self.df = new_df
        self.head = self.grow_tree(self.df)
    def grow_tree(self, df):
        if (len(df['target'].unique()) == 1):
            return df['target'].iloc[0]
        if (len(self.cols) == 0):
            modes = df['target'].mode()
            if not modes.empty:
                return modes[0]
            else:
                return 0
        best = self.best_split(df)
        node =  Node(feature=best)
        left_df = df[df[best] == 1]
        right_df = df[df[best] == 0]
        self.cols.remove(best)
        node.left = self.grow_tree(left_df)
        node.right = self.grow_tree(right_df)
        return node
    def predict(self, df):
        predictions = []
        for i, row in df.iterrows():
            temp = self.head
            while isinstance(temp, Node):
                if (row[temp.feature] == 1):
                    temp = temp.left
                else:
                    temp = temp.right
            predictions.append(temp)
        return predictions