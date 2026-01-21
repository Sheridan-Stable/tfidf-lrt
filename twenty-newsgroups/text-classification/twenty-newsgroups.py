from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups
import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from math import log, sqrt, pi
from scipy.special import gammaln
import re


def clean_text(text):
    # Lowercase
    text = text.lower()
    # Remove non-alphabetic chars (keep spaces)
    text = re.sub(r'[^a-z\s]', ' ', text)
    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def log_fact(n):
    return gammaln(n+1)


class CanonicalTfidfTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
      # Initialize scoring matrix
      d, m = X.shape
      Nj = X.sum(axis=1)
      tfidf = np.empty(shape=(d, m), dtype=np.float32)
      
      # Calculate IDF scores
      a = X.nonzero()[1]
      indices, counts = np.unique(a, return_counts=True)
      Bi = np.zeros(m)
      for i in range(len(indices)):
        index = indices[i]
        Bi[index] = counts[i]
      for i in range(m):
        if Bi[i] == 0:
          Bi[i] = 1
      IDF = np.log(d / Bi)
      
      # Calculate TF-IDF scores
      for j in range(d):
        tfidf[j] = np.multiply(X[j].toarray(), IDF)

      return sparse.csr_matrix(tfidf)
    

class LambdaTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        d, t = X.shape
        X = sparse.csr_matrix(X)
        n_j = np.array(X.sum(axis=1)).flatten()   
        n = np.sum(n_j)
        mu = 165.2860173236698 
        sigma2 = 1
        eta2 = mu**2 / sigma2
        rows, cols, data = [], [], []

        for i in range(t):  
            n_ij = X[:, i].toarray().flatten()
            n_not_ij = n_j - n_ij
            b_ij = (n_ij > 0).astype(int)

            n_i = np.sum(n_ij)
            if n_i == 0:
                continue
            b_i = np.sum(b_ij)
            n_not_i = np.sum(n_not_ij)
            r_i = n_i - b_i + 1

            for j in range(d):
                if n_ij[j] == 0:
                    continue
                # if eta2 - r_i > 0:
                tf_icf = n_ij[j] * log(n / n_i)
                tbf_idf = b_ij[j] * log(d / b_i) if b_i > 0 else 0
                correction = - log(n_ij[j]) * b_ij[j] + log_fact(n_ij[j])
                penalty = (
                    (n_ij[j] - b_ij[j]) * log((b_i + n_not_i) / (d * sigma2))
                    + n_j[j] * log(n / (b_i + n_not_i))
                    - (b_ij[j] + 1/(2*d)) * log(mu)
                    + (1/(2*d)) * (
                        (eta2 - 2*r_i + 1) * log(max(eta2 - r_i, 1)) # evaluates to 0 if eta2 - r_i <= 0
                        - (eta2 - 1.5) * log(max(eta2, 1e-9))
                        + r_i
                        - log(sqrt(2 * pi))
                    )
                )

                lam = (tf_icf - tbf_idf + correction + penalty)
                # else:
                    # lam = 0
                lam = 1 / (1 + np.exp(-lam))
                rows.append(j)
                cols.append(i)
                data.append(lam)
                
        data = np.array(data)
        return sparse.csr_matrix((data, (rows, cols)), shape=(d, t))


def main(): 
    print("Loading 20 Newsgroups dataset...")
    newsgroups_train = fetch_20newsgroups(subset='train')
    newsgroups_test = fetch_20newsgroups(subset='test')

    print("Cleaning text data...")
    X_train = [clean_text(doc) for doc in newsgroups_train.data]
    X_test = [clean_text(doc) for doc in newsgroups_test.data]
    y_train = newsgroups_train.target
    y_test = newsgroups_test.target

    print("Training and evaluating TF-IDF model...")
    text_clf = Pipeline([('vect', CountVectorizer(stop_words='english')),
                        ('tfidf', CanonicalTfidfTransformer()),
                        ('clf', MultinomialNB()),
                        ])
    text_clf.fit(X_train, y_train)
    predicted = text_clf.predict(X_test)
    print("Canonical TF-IDF Results:")
    report = metrics.classification_report(y_test, predicted)
    print(report)
    with open("../reports/tf-idf-report.txt", "w") as f:
        f.write(report)
    print("Report saved to ../reports/tf-idf-report.txt\n")


    print("Training and evaluating Lambda_i model...")
    text_clf = Pipeline([('vect', CountVectorizer(stop_words='english')),
                        ('lambda', LambdaTransformer()),
                        ('clf', MultinomialNB()),
                        ])

    text_clf.fit(X_train, y_train)
    predicted = text_clf.predict(X_test)
    print("Lambda_i Results:")
    report = metrics.classification_report(y_test, predicted)
    print(report)
    with open("../reports/lambda-i-report.txt", "w") as f:
        f.write(report)
    print("Report saved to ../reports/lambda-i-report.txt")


if __name__ == "__main__":
    main()