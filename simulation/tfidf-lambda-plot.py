import numpy as np
from math import factorial, log, sqrt, pi
from numpy.random import default_rng
import pandas as pd
from plotnine import ggplot, aes, geom_point, labs, theme_minimal

def log_fact(x):
    return log(factorial(int(x))) if x > 1 else 0.0

def main():
    rng = default_rng(42)

    # Parameters
    d = 100   # number of documents
    t = 250   # number of terms
    target_mean = 0.3

    # Gamma draws for alpha
    shape = 0.5
    scale = target_mean / shape
    alpha_true = rng.gamma(shape, scale, size=t)

    # Document sizes
    N = 82
    p = 75 / N
    n_j = rng.binomial(N, p, size=d)

    # Simulate Dirichlet-Multinomial counts
    theta = rng.dirichlet(alpha_true, size=d)
    counts = np.array([rng.multinomial(n_j[j], theta[j]) for j in range(d)])

    # Compute lambda and TF-IDF
    mu = np.mean(n_j)
    sigma2 = 1.0
    eta2 = mu**2 / sigma2

    lambda_vals = []
    sum_tfidf_vals = []

    for i in range(t):
        n_ij = counts[:, i]
        n_not_ij = np.sum(counts, axis=1) - n_ij
        b_ij = (n_ij > 0).astype(int)

        n_i = np.sum(n_ij)
        b_i = np.sum(b_ij)
        n_not_i = np.sum(n_not_ij)
        r_i = n_i - b_i + 1

        # Per-doc TF-IDF for term i
        tfidf = [n_ij[j] * log(d / b_i) if n_ij[j] > 0 else 0 for j in range(d)]

        # Per-doc lambda summand for term i
        lambda_summand = []
        for j in range(d):
            tf_icf = n_ij[j] * log(np.sum(n_j) / n_i) if n_ij[j] > 0 else 0
            tbf_idf = b_ij[j] * log(d / b_i) if b_ij[j] > 0 else 0
            correction = -log(n_ij[j]) * b_ij[j] + log_fact(n_ij[j]) if n_ij[j] > 0 else 0
            penalty = (
                (n_ij[j] - b_ij[j]) * log((b_i + n_not_i) / (d * sigma2))
                + n_j[j] * log(np.sum(n_j) / (b_i + n_not_i))
                - (b_ij[j] + 1/(2*d)) * log(mu)
                + (1/(2*d)) * (
                    (eta2 - 2*r_i + 1) * log(eta2 - r_i)
                    - (eta2 - 1.5) * log(eta2)
                    + r_i
                    # - log(sqrt(2 * pi)) removed constant term
                )
            )
            lambda_summand.append(2 * (tf_icf - tbf_idf + correction + penalty))

        lambda_vals.append(np.sum(lambda_summand))
        sum_tfidf_vals.append(np.sum(tfidf))

    # Scatter plot
    df_scatter = pd.DataFrame({
        'sum_tfidf_vals': sum_tfidf_vals,
        'lambda_vals': lambda_vals,
        'alpha_size': alpha_true
    })

    print("Generating scatter plot of lambda_i vs. total TF-IDF of Figure 3 in the manuscript...")
    p = (
        ggplot(df_scatter, aes(x='sum_tfidf_vals', y='lambda_vals', size='alpha_size'))
        + geom_point(color='#619CFF', alpha=0.9, stroke=0)
        + labs(
            x='Total TF–IDF across documents',
            y=r'$\lambda_i$ score',
            title=r'Relationship between $\lambda_i$ and total TF–IDF score',
            size=r'$\alpha_i$'
        )
        + theme_minimal()
    )

    print("Saving scatter plot to 'lambda_tfidf.pdf'...")
    p.save("./plots/lambda_tfidf.pdf", dpi=300)
    print("Scatter plot saved.")
    corr = np.corrcoef(sum_tfidf_vals, lambda_vals)[0, 1]
    print("\nCorrelation between lambda_i and total TF-IDF:", corr)
    with open("./reports/tfidf-lambda-correlation.txt", "w") as f:
        f.write(f"Correlation between lambda_i and total TF-IDF: {corr}")


if __name__ == "__main__":
    main()
