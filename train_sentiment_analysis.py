import pickle
import random
import re
from random import shuffle
from string import punctuation

from nltk import download
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy
from nltk.corpus import movie_reviews
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.util import everygrams
from nltk.tokenize import word_tokenize

download('movie_reviews')
download('punkt')
download('stopwords')
download('wordnet')

stopwords_eng = stopwords.words('english')


def bag_of_words(document):
    bag = {}
    for w in document:
        bag[w] = bag.get(w, 0) + 1
    return bag


lemmatizer = WordNetLemmatizer()


def extract_feature(document):
    words = word_tokenize(document)
    lemmas = [str(lemmatizer.lemmatize(w)) for w in words if w not in stopwords_eng and w not in punctuation]
    document = " ".join(lemmas)
    document = document.lower()
    document = re.sub(r'[^a-zA-Z0-9\s]', ' ', document)
    words = [w for w in document.split(" ") if w != "" and w not in stopwords_eng and w not in punctuation]
    return [str(" ".join(ngram)) for ngram in list(everygrams(words, max_len=3))]


reviews_pos = []
reviews_neg = []
for fileid in movie_reviews.fileids('pos'):
    words = extract_feature(movie_reviews.raw(fileid))
    reviews_pos.append((bag_of_words(words), 'pos'))
for fileid in movie_reviews.fileids('neg'):
    words = extract_feature(movie_reviews.raw(fileid))
    reviews_neg.append((bag_of_words(words), 'neg'))

split_pct = .80


def split_set(review_set):
    split = int(len(review_set) * split_pct)
    return review_set[:split], review_set[split:]


random.seed(10)
shuffle(reviews_pos)
shuffle(reviews_neg)

pos_train, pos_test = split_set(reviews_pos)
neg_train, neg_test = split_set(reviews_neg)

train_set = pos_train + neg_train
test_set = pos_test + neg_test

model = NaiveBayesClassifier.train(train_set)

print(100 * accuracy(model, test_set))

model_file = open('sa_classifier.pickle', 'wb')
pickle.dump(model, model_file)
model_file.close()
print('saved')
