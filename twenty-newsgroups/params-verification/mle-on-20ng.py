import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from scipy.optimize import minimize
from scipy.special import gammaln
import re
import time


def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def beta_binomial_neg_log_likelihood(params, nij, nj):
    alpha_i, alpha_not_i = params
    log_B_num = gammaln(nij + alpha_i) + gammaln(nj - nij + alpha_not_i) - gammaln(nj + alpha_i + alpha_not_i)
    log_B_den = gammaln(alpha_i) + gammaln(alpha_not_i) - gammaln(alpha_i + alpha_not_i)
    log_likelihood = np.sum(log_B_num - log_B_den)
    return -log_likelihood


def fit_beta_binomial(term_counts, total_counts):
    nij = np.array(term_counts)
    nj = np.array(total_counts)
    
    mask = nj > 0
    nij = nij[mask]
    nj = nj[mask]
    
    if len(nij) == 0:
        return 0.0, 0.0

    # Initializing with uniform prior for MLE
    init_alpha_i = 1.0
    init_alpha_not_i = 1.0
    
    result = minimize(
        beta_binomial_neg_log_likelihood,
        x0=[init_alpha_i, init_alpha_not_i],
        args=(nij, nj),
        bounds=((1e-6, None), (1e-6, None)),
        method='L-BFGS-B'
    )
    return result.x[0], result.x[1]

def main():
    print("Loading 20 Newsgroups dataset...")
    newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
    
    print("Cleaning text data...")
    data = [clean_text(doc) for doc in newsgroups_train.data]

    print("Vectorizing...")
    vectorizer = CountVectorizer() 
    X = vectorizer.fit_transform(data)
    vocab = vectorizer.get_feature_names_out()
    
    doc_lengths = np.array(X.sum(axis=1)).flatten()
    
    print("Fitting beta-binomial models (this may take a while)...")
    
    alphas_i = []
    alphas_not_i = []
    
    start_time = time.time()
    for i, term in enumerate(vocab):
        if i % 100 == 0 and i > 0:
            elapsed = time.time() - start_time
            rate = i / elapsed
            print(f"Processed {i}/{len(vocab)} terms. Est. remaining: {(len(vocab)-i)/rate/60:.1f} min", end='\r')
            
        term_counts = X[:, i].toarray().flatten()
        a_i, a_not_i = fit_beta_binomial(term_counts, doc_lengths)
        alphas_i.append(a_i)
        alphas_not_i.append(a_not_i)
        
    print(f"\nFinished fitting in {(time.time() - start_time)/60:.1f} minutes.")
    
    alphas_i = np.array(alphas_i)
    alphas_not_i = np.array(alphas_not_i)
    
    # Store CSV
    print("Saving results...")

    df = pd.DataFrame({
        'term': vocab,
        'alpha_i': alphas_i,
        'alpha_not_i': alphas_not_i
    })
    df.to_csv('../reports/20ng-bb-params-mle.csv', index=False)
    
if __name__ == "__main__":
    main()
