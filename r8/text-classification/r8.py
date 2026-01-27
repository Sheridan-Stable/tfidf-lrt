from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.special import gammaln
import re
from nltk.corpus import reuters


def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def log_fact(n):
    return gammaln(n + 1)


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
    def __init__(self, mu=119, sigma2=1.0):
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


def load_r8():
    r8_classes = [
        'acq', 'crude', 'earn', 'grain',
        'interest', 'money-fx', 'ship', 'trade'
    ]

    def is_r8(doc_id):
        labels = reuters.categories(doc_id)
        return len(labels) == 1 and labels[0] in r8_classes

    all_ids = reuters.fileids()

    train_ids = [
        d for d in all_ids
        if d.startswith('training/') and is_r8(d)
    ]

    test_ids = [
        d for d in all_ids
        if d.startswith('test/') and is_r8(d)
    ]

    X_train = [clean_text(reuters.raw(d)) for d in train_ids]
    X_test  = [clean_text(reuters.raw(d)) for d in test_ids]

    label_map = {label: i for i, label in enumerate(r8_classes)}

    y_train = np.array([label_map[reuters.categories(d)[0]] for d in train_ids])
    y_test  = np.array([label_map[reuters.categories(d)[0]] for d in test_ids])

    return X_train, X_test, y_train, y_test, r8_classes


def main():
    print("Loading R8 dataset...")
    X_train, X_test, y_train, y_test, target_names = load_r8()

    print("Training and evaluating Canonical TF-IDF model...")
    text_clf = Pipeline([
        ('vect', CountVectorizer(stop_words='english')),
        ('tfidf', CanonicalTfidfTransformer()),
        ('clf', MultinomialNB()),
    ])

    text_clf.fit(X_train, y_train)
    predicted = text_clf.predict(X_test)

    report = metrics.classification_report(
        y_test, predicted, target_names=target_names
    )
    print(report)

    with open("../reports/r8-tfidf-report.txt", "w") as f:
        f.write(report)

    print("Training and evaluating Lambda_i model...")
    text_clf = Pipeline([
        ('vect', CountVectorizer(stop_words='english')),
        ('lambda', LambdaTransformer()),
        ('clf', MultinomialNB()),
    ])

    text_clf.fit(X_train, y_train)
    predicted = text_clf.predict(X_test)

    report = metrics.classification_report(
        y_test, predicted, target_names=target_names
    )
    print(report)

    with open("../reports/r8-lambda-i-report.txt", "w") as f:
        f.write(report)

if __name__ == "__main__":
    main()
