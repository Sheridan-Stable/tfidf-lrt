import pandas as pd
import numpy as np
# fix seed 
np.random.seed(42)
df = pd.read_csv("../reports/r8-bb-params-mle.csv")

# find prop of words with alpha_i < 1
p_alpha_i_lt_1 = len(df[df['alpha_i'] < 1]) / len(df)
print("Prop of words with alpha_i < 1:", p_alpha_i_lt_1)

# find prop of words with alpha_not_i > 10
p_alpha_not_i_gt_10 = len(df[df['alpha_not_i'] > 10]) / len(df)
print("Prop of words with alpha_not_i > 10:", p_alpha_not_i_gt_10)

# print representative words with alpha_i < 1 and alpha_not_i > 10
print("Representative words with alpha_i < 1 and alpha_not_i > 10:")
print(df[(df['alpha_i'] < 1) & (df['alpha_not_i'] > 10)][['term', 'alpha_i', 'alpha_not_i']].sample(20))

# find prop of words with alpha_i > 1 or alpha_not_i < 10
p_alpha_i_gt_1_or_alpha_not_i_lt_10 = len(df[(df['alpha_i'] > 1) | (df['alpha_not_i'] < 10)]) / len(df)
print("Prop of words with alpha_i > 1 or alpha_not_i < 10:", p_alpha_i_gt_1_or_alpha_not_i_lt_10)

# print other representative words with the other way around (OR) and print their alphas  
print("Representative words with alpha_i > 1 or alpha_not_i < 10:")
print(df[(df['alpha_i'] > 1) | (df['alpha_not_i'] < 10)][['term', 'alpha_i', 'alpha_not_i']].sample(20))
