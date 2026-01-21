import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.special import gammaln
import re
from math import log, sqrt, pi

from nltk.corpus import reuters

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

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def log_fact(n):
    return gammaln(n + 1)
class LambdaTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, mu=119, sigma2=1.0):
        self.mu = mu
        self.sigma2 = sigma2

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        fallback = 0
        d, t = X.shape
        X = sparse.csr_matrix(X)
        n_j = np.array(X.sum(axis=1)).flatten()
        n = np.sum(n_j)

        mu = self.mu
        sigma2 = self.sigma2
        eta2 = mu ** 2 / sigma2

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

                if eta2 - r_i > 0:
                    tf_icf = n_ij[j] * log(n / n_i)
                    tbf_idf = b_ij[j] * log(d / b_i) if b_i > 0 else 0
                    correction = -log(n_ij[j]) * b_ij[j] + log_fact(n_ij[j])

                    penalty = (
                        (n_ij[j] - b_ij[j]) * log((b_i + n_not_i) / (d * sigma2))
                        + n_j[j] * log(n / (b_i + n_not_i))
                            - (b_ij[j] + 1 / (2 * d)) * log(mu)
                            + (1 / (2 * d)) * (
                                (eta2 - 2 * r_i + 1) * log(eta2 - r_i)
                                - (eta2 - 1.5) * log(max(eta2, 1e-9))
                                + r_i
                                - log(sqrt(2 * pi))
                            )
                        )

                    lam = tf_icf - tbf_idf + correction + penalty
                else:
                    lam = 0
                    fallback += 1

                lam = 1 / (1 + np.exp(-lam))
                rows.append(j)
                cols.append(i)
                data.append(lam)

        print(f"Fallback: {fallback}")
        return sparse.csr_matrix((data, (rows, cols)), shape=(d, t))

def main():
    X_train_raw, X_test_raw, y_train, y_test, _ = load_r8()
    
    mu_values = [150, 250, 500, 1000, 2000]
    sigma_values = [0.2, 0.4, 0.6, 0.8, 1.0]
    results = np.zeros((len(sigma_values), len(mu_values)))

    # Vectorize text once
    vect = CountVectorizer(stop_words='english')
    X_train_counts = vect.fit_transform(X_train_raw)
    X_test_counts = vect.transform(X_test_raw)

    print("Generating sensitivity analysis grid...")
    for i, s2 in enumerate(sigma_values[::-1]):
        for j, m in enumerate(mu_values):
            lt = LambdaTransformer(mu=m, sigma2=s2)
            X_train_lam = lt.transform(X_train_counts)
            X_test_lam = lt.transform(X_test_counts)
            
            clf = MultinomialNB()
            clf.fit(X_train_lam, y_train)
            acc = metrics.accuracy_score(y_test, clf.predict(X_test_lam))
            
            results[i, j] = acc
            print(f"Sigma2: {s2:.2f}, Mu: {m} -> Accuracy: {acc:.6f}")

    plt.figure(figsize=(10, 6))
    sns.heatmap(results, annot=True, fmt=".5f", 
                xticklabels=mu_values, yticklabels=sigma_values[::-1], cmap="YlGnBu", cbar=False)
    plt.title("Accuracy across $\mu$ and $\sigma^2$")
    plt.xlabel("$\mu$")
    plt.ylabel("$\sigma^2$")
    plt.savefig("r8-sensitivity-mag.pdf", transparent=True, bbox_inches='tight')

if __name__ == "__main__":
    main()