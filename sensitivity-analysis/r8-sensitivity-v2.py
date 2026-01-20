from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from math import log, sqrt, pi
from scipy.special import gammaln
import re
from nltk.corpus import reuters
import matplotlib.pyplot as plt
import seaborn as sns


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


import numpy as np
from math import log, sqrt, pi
from scipy.special import gammaln

def log_fact(x):
    return gammaln(x + 1)

class LambdaTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, mu=119, sigma2=1.0):
        self.mu = mu
        self.sigma2 = sigma2

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = sparse.csr_matrix(X)
        d, t = X.shape
        n_j = np.array(X.sum(axis=1)).flatten()  # sum over rows
        n = np.sum(n_j)

        mu = self.mu
        sigma2 = self.sigma2
        eta2 = mu ** 2 / sigma2

        rows, cols, data = [], [], []
        fallback_terms = 0

        # Precompute sums over terms
        n_i_all = np.array(X.sum(axis=0)).flatten()
        b_i_all = np.array((X > 0).sum(axis=0)).flatten()
        n_not_i_all = np.sum(n_j) - n_i_all
        r_i_all = n_i_all - b_i_all + 1

        for i in range(t):
            n_ij = X[:, i].toarray().flatten()
            n_not_ij = n_j - n_ij
            b_ij = (n_ij > 0).astype(int)

            n_i = n_i_all[i]
            b_i = b_i_all[i]
            n_not_i = n_not_i_all[i]
            r_i = r_i_all[i]

            if n_i == 0 or (eta2 - r_i <= 0):
                fallback_terms += 1
                continue

            # vectorized per term across all documents
            tf_icf = n_ij * np.log(n / n_i)
            tbf_idf = b_ij * np.log(d / b_i) if b_i > 0 else 0
            correction = -np.log(n_ij, where=(b_ij > 0)) * b_ij + log_fact(n_ij)

            penalty = ((n_ij - b_ij) * np.log((b_i + n_not_i) / (d * sigma2))
                       + n_j * np.log(n / (b_i + n_not_i))
                       - (b_ij + 1 / (2 * d)) * np.log(mu)
                       + (1 / (2 * d)) * ((eta2 - 2 * r_i + 1) * np.log(eta2 - r_i)
                                          - (eta2 - 1.5) * np.log(max(eta2, 1e-9))
                                          + r_i - np.log(sqrt(2 * pi))))

            lam = tf_icf - tbf_idf + correction + penalty
            lam = 1 / (1 + np.exp(-lam))

            nonzero_idx = np.nonzero(n_ij)[0]
            rows.extend(nonzero_idx)
            cols.extend([i] * len(nonzero_idx))
            data.extend(lam[nonzero_idx])

        print(f"Fallback: {fallback_terms} out of {t} unique terms")
        return sparse.csr_matrix((data, (rows, cols)), shape=(d, t))


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
    X_train_raw, X_test_raw, y_train, y_test, _ = load_r8()
    
    mu_values = [50, 125, 250, 500]
    sigma_values = [0.25, 0.5, 1.00, 2.00]
    results = np.zeros((len(sigma_values), len(mu_values)))

    # Vectorize text once
    vect = CountVectorizer(stop_words='english')
    X_train_counts = vect.fit_transform(X_train_raw)
    X_test_counts = vect.transform(X_test_raw)

    print("Starting Sensitivity Analysis Grid...")
    for i, s2 in enumerate(sigma_values[::-1]):
        for j, m in enumerate(mu_values):
            lt = LambdaTransformer(mu=m, sigma2=s2)
            X_train_lam = lt.transform(X_train_counts)
            X_test_lam = lt.transform(X_test_counts)
            
            clf = MultinomialNB()
            clf.fit(X_train_lam, y_train)
            acc = metrics.accuracy_score(y_test, clf.predict(X_test_lam))
            
            results[i, j] = acc
            print(f"Sigma2: {s2:.2f}, Mu: {m} -> Accuracy: {acc:.4f}")

    # Plotting Heatmap and save it
    plt.figure(figsize=(10, 6))
    sns.heatmap(results, annot=True, fmt=".4f", 
                xticklabels=mu_values, yticklabels=sigma_values[::-1], cmap="YlGnBu")
    plt.title("Sensitivity Analysis: Accuracy across $\mu$ and $\sigma^2$")
    plt.xlabel("$\mu$")
    plt.ylabel("$\sigma^2$")
    plt.show()
    plt.savefig("r8-sensitivity.png")

if __name__ == "__main__":
    main()