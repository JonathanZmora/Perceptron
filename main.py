import pandas as pd
from Perceptron import Perceptron
pd.options.mode.chained_assignment = None

df = pd.read_csv('Processed Wisconsin Diagnostic Breast Cancer.csv')

# replacing 0 with -1 in the 'diagnosis' column
df['diagnosis'] = df['diagnosis'].apply(lambda x: 1 if x == 1 else -1)

# shuffling the data
df = df.sample(frac=1)

X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

# splitting data into train (80%) and test (20%)
split_index = int(X.shape[0] * 0.8)
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

# training the model
perceptron = Perceptron()
perceptron.fit(X_train, y_train)

# computing train and test scores
train_score = perceptron.train_score
test_score = perceptron.score(X_test, y_test)

print(f'train error: {1-train_score:.3f}, test error: {1-test_score:.3f}')
