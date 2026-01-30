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
    X_train_raw, X_test_raw, y_train, y_test, _ = load_r8()
    
    mu_values = [30, 60, 90, 120, 150, 180]
    sigma_values = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
    results = np.zeros((len(sigma_values), len(mu_values)))

    # Vectorize text once
    vect = CountVectorizer(stop_words='english')
    X_train_counts = vect.fit_transform(X_train_raw)
    X_test_counts = vect.transform(X_test_raw)

    print(f"Mean document length: {X_train_counts.sum(axis=1).mean():.2f}")
    print(f"Vocabulary size of training set: {X_train_counts.shape[1]}")
    print(f"Number of documents in training set: {X_train_counts.shape[0]}")
    print(f"Vocabulary size of test set: {X_test_counts.shape[1]}")
    print(f"Number of documents in test set: {X_test_counts.shape[0]}")

    print("Generating sensitivity analysis grid...")
    for i, s2 in enumerate(sigma_values[::-1]):
        for j, m in enumerate(mu_values):
            lt = LambdaTransformer(mu=m, sigma2=s2)
            X_train_lam = lt.fit_transform(X_train_counts)
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
    plt.savefig("../plots/r8-sensitivity-analysis-grid.pdf", transparent=True, bbox_inches='tight')

if __name__ == "__main__":
    main()