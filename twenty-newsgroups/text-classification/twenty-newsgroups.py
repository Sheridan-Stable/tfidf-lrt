from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups
import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
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
        d, m = X.shape
        tfidf = np.empty((d, m), dtype=np.float32)

        # document frequency
        a = X.nonzero()[1]
        indices, counts = np.unique(a, return_counts=True)
        Bi = np.ones(m)
        Bi[indices] = counts

        IDF = np.log(d / Bi)

        for j in range(d):
            tfidf[j] = np.multiply(X[j].toarray(), IDF)

        return sparse.csr_matrix(tfidf)

class LambdaTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, mu, sigma2=1.0):
        self.mu = mu
        self.sigma2 = sigma2

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.tocsr()
        d, t = X.shape

        # Document stats
        n_j = X.sum(axis=1).A1
        n = n_j.sum()

        # Term stats
        n_i = X.sum(axis=0).A1
        b_i = (X > 0).sum(axis=0).A1
        n_not_i = n - n_i
        r_i = n_i - b_i + 1

        mu = self.mu
        sigma2 = self.sigma2
        eta2 = mu ** 2 / sigma2

        rows, cols = X.nonzero()
        n_ij = X.data

        # Vectorized components
        tf_icf = n_ij * np.log(n / n_i[cols])
        tbf_idf = np.log(d / b_i[cols])

        correction = -np.log(n_ij) + gammaln(n_ij + 1)

        penalty = (
            (n_ij - 1) * np.log((b_i[cols] + n_not_i[cols]) / (d * sigma2))
            + n_j[rows] * np.log(n / (b_i[cols] + n_not_i[cols]))
            - (1 + 1 / (2 * d)) * np.log(mu)
            + (1 / (2 * d)) * (
                (eta2 - 2 * r_i[cols] + 1)
                * np.log(np.maximum(eta2 - r_i[cols], 1))
                - (eta2 - 1.5) * np.log(np.maximum(eta2, 1e-9))
                + r_i[cols]
            )
        )

        lam = tf_icf - tbf_idf + correction + penalty
        lam = 1 / (1 + np.exp(-lam))

        return sparse.csr_matrix((lam, (rows, cols)), shape=(d, t))

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

    # Vectorize to get mean document length
    vect = CountVectorizer(stop_words='english')
    X_train_counts = vect.fit_transform(X_train)
    mu_mean_doc_length = X_train_counts.sum(axis=1).mean()
    print(f"Mean document length (train): {mu_mean_doc_length:.2f}")

    print("Training and evaluating Lambda_i model...")
    text_clf = Pipeline([('vect', vect),
                        ('lambda', LambdaTransformer(mu=mu_mean_doc_length)),
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