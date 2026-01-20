import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from scipy.optimize import minimize
from scipy.special import gammaln
import re
import time
from nltk.corpus import reuters

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


def load_r8():
    r8_classes = ['acq', 'crude', 'earn', 'grain', 'interest', 'money-fx', 'ship', 'trade']
    def is_r8(doc_id):
        labels = reuters.categories(doc_id)
        return len(labels) == 1 and labels[0] in r8_classes

    all_ids = reuters.fileids()
    train_ids = [d for d in all_ids if d.startswith('training/') and is_r8(d)]
    
    # We use the training set for parameter verification
    X_train = [clean_text(reuters.raw(d)) for d in train_ids]
    return X_train

def main():
    print("Loading R8 dataset...")
    data = load_r8()

    print("Vectorizing...")
    # min_df=5
    vectorizer = CountVectorizer(stop_words='english', min_df=5) 
    X = vectorizer.fit_transform(data)
    vocab = vectorizer.get_feature_names_out()
    doc_lengths = np.array(X.sum(axis=1)).flatten()
    
    print(f"Vocabulary size: {len(vocab)}")
    print("Fitting Beta-Binomial models for R8 terms...")
    
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

    # Save CSV
    df = pd.DataFrame({'term': vocab, 'alpha_i': alphas_i, 'alpha_not_i': alphas_not_i})
    df.to_csv('r8_bb_params.csv', index=False)
    
    # Visualization
    print("Generating plots...")
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(np.log(alphas_i), bins=50, color='skyblue', alpha=0.7, label='log(alpha_i)')
    plt.hist(np.log(alphas_not_i), bins=50, color='orange', alpha=0.7, label='log(alpha_not_i)')
    plt.xlabel('Log parameter value')
    plt.ylabel('Frequency')
    plt.title('R8: Distribution of beta-binomial parameters')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('r8_beta_binomial_verification.png')
    print("Saved results to r8_bb_params.csv and r8_beta_binomial_verification.png")

if __name__ == "__main__":
    main()