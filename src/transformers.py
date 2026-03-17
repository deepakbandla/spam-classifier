from sklearn.base import BaseEstimator, TransformerMixin
from collections import Counter
import numpy as np
import re
import nltk
from preprocessing import email_to_text

stemmer = nltk.PorterStemmer()

class EmailToWordCounterTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, lower_case=True):
        self.lower_case = lower_case

    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X_transformed = []

        for email in X:
            text = email_to_text(email) or ""

            if self.lower_case:
                text = text.lower()

            text = re.sub(r'\W+', ' ', text)
            word_counts = Counter(text.split())
            stemmed_word_counts = Counter()

            for word, count in word_counts.items():
                stemmed = stemmer.stem(word)
                stemmed_word_counts[stemmed] += count

            X_transformed.append(stemmed_word_counts)

        return np.array(X_transformed)
    
from scipy.sparse import csr_matrix

class WordCounterToVectorTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, vocabulary_size=1000):
        self.vocabulary_size = vocabulary_size
    def fit(self, X, y=None):
        total_count = Counter()
        for word_count in X:
            for word, count in word_count.items():
                total_count[word] += min(count, 10)

        most_common = total_count.most_common()[:self.vocabulary_size]

        self.vocabulary_ = {word: index + 1 for index, (word, count) in enumerate(most_common)}

        return self

    def transform(self, X, y=None):
        rows, cols, data = [], [], []
        for row, word_count in enumerate(X):
            for word, count in word_count.items():
                rows.append(row)
                cols.append(self.vocabulary_.get(word, 0))
                data.append(count)

        return csr_matrix(
            (data, (rows, cols)), shape=(len(X), self.vocabulary_size + 1)
        )