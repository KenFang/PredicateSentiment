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

download('movie_reviews')
download('punkt')
download('stopwords')
download('wordnet')

stopwords_eng = stopwords.words('english')


def extract_features(movie_review_words):
    return [w for w in movie_review_words if w not in stopwords_eng and w not in punctuation]


def bag_of_words(feature_words):
    bag = {}
    for w in feature_words:
        bag[w] = bag.get(w, 0) + 1
    return bag


reviews_pos = []
reviews_neg = []
for fileid in movie_reviews.fileids('pos'):
    words = extract_features(movie_reviews.words(fileid))
    reviews_pos.append((bag_of_words(words), 'pos'))

for fileid in movie_reviews.fileids('neg'):
    words = extract_features(movie_reviews.words(fileid))
    reviews_neg.append((bag_of_words(words), 'neg'))

split_pct = .80


def split_set(review_set):
    split = int(len(review_set) * split_pct)
    return review_set[:split], review_set[split:]


random.seed(0)
shuffle(reviews_pos)
shuffle(reviews_neg)

pos_train, pos_test = split_set(reviews_pos)
neg_train, neg_test = split_set(reviews_neg)

train_set = pos_train + neg_train
test_set = pos_test + neg_test

model = NaiveBayesClassifier.train(train_set)

most_informative_feature = model.most_informative_features(500)

most_informative_words = []
for i in range(len(most_informative_feature)):
    most_informative_words.append(most_informative_feature[i][0])


def generate_ngrams(s, n=1):
    # Convert to lowercase
    s = s.lower()

    # Replace all non alphanumeric characters with space
    s = re.sub(r'[^a-zA-Z0-9\s]', ' ', s)

    # Break sentence in the token, remove empty tokens
    tokens = [token for token in s.split(" ") if token != ""]

    ngrams_words = zip(*[tokens[i:] for i in range(n)])
    return [" ".join(ngram) for ngram in ngrams_words]


informative_lemmatizer = WordNetLemmatizer()


def extract_informative_feature(reviews_words):
    return [informative_lemmatizer.lemmatize(w) for w in reviews_words if w in most_informative_words]


ngram_review_pos = []
ngram_review_neg = []
for fileid in movie_reviews.fileids('pos'):
    ngram_pos = generate_ngrams(movie_reviews.raw(fileid))
    ngram_words_pos = extract_informative_feature(ngram_pos)
    ngram_review_pos.append((bag_of_words(ngram_words_pos), 'pos'))
for fileid in movie_reviews.fileids('neg'):
    ngram_neg = generate_ngrams(movie_reviews.raw(fileid))
    ngram_words_neg = extract_informative_feature(ngram_neg)
    ngram_review_neg.append((bag_of_words(ngram_words_neg), 'neg'))

random.seed(10)
shuffle(ngram_review_pos)
shuffle(ngram_review_neg)

ngram_pos_train, ngram_pos_test = split_set(ngram_review_pos)
ngram_neg_train, ngram_neg_test = split_set(ngram_review_neg)

ngram_train_set = ngram_pos_train + ngram_neg_train
ngram_test_set = ngram_pos_test + ngram_neg_test

ngram_review_model = NaiveBayesClassifier.train(ngram_train_set)

print(100 * accuracy(ngram_review_model, ngram_test_set))

model_file = open('sa_classifier.pickle', 'wb')
pickle.dump(model, model_file)
model_file.close()
