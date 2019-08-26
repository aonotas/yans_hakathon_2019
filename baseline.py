import csv
from gensim.utils import tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from sklearn.svm import LinearSVC

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
# questions = [' '.join(list(tokenize(row[0], to_lower=True))) for row in lines] # tokenize
questions1 = [' '.join(list(tokenize(row[0], to_lower=True))) for row in train] # tokenize
questions2 = [' '.join(list(tokenize(row[1], to_lower=True))) for row in train] # tokenize
# questions1 = [row[0] for row in train] # no tokenize
# questions2 = [row[1] for row in train] # no tokenize
y = [int(row[-1]) for row in train]

# tf-idf classifior
vectorizer = TfidfVectorizer()
questions = questions1 + questions2
vectorizer.fit(questions)
X1 = vectorizer.transform(questions1)
X2 = vectorizer.transform(questions2)

print('features:', vectorizer.get_feature_names())

X = hstack((X1, X2))
print('X:', X.shape)

# train classifier
print('train classifier:')
clf = LinearSVC().fit(X, y)

# predict
dev = load_dataset(filename='dev.csv')
questions1 = [' '.join(list(tokenize(row[0], to_lower=True))) for row in dev] # tokenize
questions2 = [' '.join(list(tokenize(row[1], to_lower=True))) for row in dev] # tokenize
y_dev_gold = [int(row[-1]) for row in dev]
X1 = vectorizer.transform(questions1)
X2 = vectorizer.transform(questions2)
X_dev = hstack((X1, X2))
print('X_dev:', X_dev.shape)
y_dev_predict = clf.predict(X_dev)
print('y_dev_predict:', y_dev_predict)
print('y_dev_gold:', y_dev_gold)

result = [predict == gold for predict, gold in zip(y_dev_predict, y_dev_gold)]

print('acuracy [dev]:', 100.0 * sum(result) / len(result))
