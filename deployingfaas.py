import pickle
import re
from string import punctuation

from nltk import download
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.util import everygrams

download('punkt')
download('stopwords')
download('wordnet')

stopwords_eng = stopwords.words('english')

lemmatizer = WordNetLemmatizer()


def extract_feature(document):
    words = word_tokenize(document)
    lemmas = [str(lemmatizer.lemmatize(w)) for w in words if w not in stopwords_eng and w not in punctuation]
    document = " ".join(lemmas)
    document = document.lower()
    document = re.sub(r'[^a-zA-Z0-9\s]', ' ', document)
    words = [w for w in document.split(" ") if w != "" and w not in stopwords_eng and w not in punctuation]
    return [str(" ".join(ngram)) for ngram in list(everygrams(words, max_len=3))]


def bag_of_words(document):
    bag = {}
    for w in document:
        bag[w] = bag.get(w, 0) + 1
    return bag


model_file = open('sa_classifier.pickle', 'rb')
model = pickle.load(model_file)
model_file.close()


def get_sentiment(review):
    words = extract_feature(review)
    words = bag_of_words(words)
    return model.classify(words)


positive_review = 'This movie is amazing, with witty dialog and beautiful shots.'
print('positive_review: ' + get_sentiment(positive_review))

negative_review = 'I hated everything about this unimaginative mess. Two thumbs down.'
print('negative_review: ' + get_sentiment(negative_review))

positive_negative_review = 'I hated This movie is amazing, with witty dialog and beautiful shots. Two thumbs down.'
print('positive ? negative ? : ' + get_sentiment(positive_negative_review))
