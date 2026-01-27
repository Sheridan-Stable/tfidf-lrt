from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import re
from nltk.corpus import reuters
import matplotlib.pyplot as plt


def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def load_reuters():
    docs = reuters.fileids()
    X = [clean_text(reuters.raw(d)) for d in docs]
    y = [reuters.categories(d)[0] for d in docs]
    return X, np.array(y)


def main():
    X, _ = load_reuters()

    vectorizer = CountVectorizer()
    X_counts = vectorizer.fit_transform(X)

    # all nonzero n_ij values
    all_counts = X_counts.data
    all_counts.sort()

    # save CSV
    np.savetxt(
        "./reports/word-stats-reuters.csv",
        all_counts,
        delimiter=",",
        header="count",
        comments=""
    )

    # plot histogram and remove outliers for visualization
    all_counts = all_counts[all_counts < np.percentile(all_counts,99.5)]
    plt.hist(all_counts, bins=50)
    plt.xlabel("Count")
    plt.ylabel("Frequency")
    plt.title("Distribution of word counts in Reuters dataset")
    plt.show()
    plt.savefig("./plots/word-stats-reuters.png")

    print("Proportion of words with count <= 7:", np.mean(all_counts <= 7))


if __name__ == "__main__":
    main()
