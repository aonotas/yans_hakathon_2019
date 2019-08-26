import csv
from gensim.utils import tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
import numpy as np

# load glove vectors
import gensim.downloader as api
model = api.load("glove-wiki-gigaword-50")
# https://github.com/RaRe-Technologies/gensim-data#models
# model.most_similar("what")


# load dataset

def load_dataset(filename):
    f = open(filename, 'r')
    reader = csv.reader(f)
    header = next(reader)
    lines = [row for row in reader]
    f.close()
    return lines

train = load_dataset(filename='train_100000.csv')
print('#train:', len(train))
questions1 = [list(tokenize(row[0], to_lower=True)) for row in train] # tokenize
questions2 = [list(tokenize(row[1], to_lower=True)) for row in train] # tokenize
# questions1 = [row[0] for row in train] # no tokenize
# questions2 = [row[1] for row in train] # no tokenize
y = [int(row[-1]) for row in train]

# glove vectors
# print(model["what"].shape)
def _to_vec(words, model):
    if len([w for w in words if w in model]):
        return np.sum([model[w] for w in words if w in model], axis=0).reshape(1, 50)
    else:
        return np.zeros((1, 50))

X1 = np.concatenate([_to_vec(words, model) for words in questions1], axis=0)
X2 = np.concatenate([_to_vec(words, model) for words in questions2], axis=0)
print('X1:', X1.shape)
print('X2:', X2.shape)
# vectorizer = TfidfVectorizer()
# questions = questions1 + questions2
# vectorizer.fit(questions)
# X1 = vectorizer.transform(questions1)
# X2 = vectorizer.transform(questions2)

# print('features:', vectorizer.get_feature_names())

X = np.concatenate([X1, X2], axis=1)
print('X:', X.shape)

# train classifier
print('train classifier:')
clf = MLPClassifier().fit(X, y)

# predict
dev = load_dataset(filename='dev.csv')

questions1 = [list(tokenize(row[0], to_lower=True)) for row in dev] # tokenize
questions2 = [list(tokenize(row[1], to_lower=True)) for row in dev] # tokenize
y_dev_gold = [int(row[-1]) for row in dev]
X1 = np.concatenate([_to_vec(words, model) for words in questions1], axis=0)
X2 = np.concatenate([_to_vec(words, model) for words in questions2], axis=0)
X_dev = np.concatenate([X1, X2], axis=1)
print('X_dev:', X_dev.shape)
y_dev_predict = clf.predict(X_dev)
print('y_dev_predict:', y_dev_predict)
print('y_dev_gold:', y_dev_gold)

result = [predict == gold for predict, gold in zip(y_dev_predict, y_dev_gold)]

print('acuracy [dev]:', 100.0 * sum(result) / len(result))
