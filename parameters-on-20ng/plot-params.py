import numpy as np
import pandas as pd
from plotnine import (
    ggplot, aes, geom_histogram, labs, theme_minimal, 
    scale_fill_manual
)

def main():
    try:
        df = pd.read_csv("bb_params_verification.csv")
    except FileNotFoundError:
        print("Error: 'bb_params_verification.csv' not found.")
        return

    df['log_alpha_i'] = np.log(df['alpha_i'])
    df['log_alpha_not_i'] = np.log(df['alpha_not_i'])

    df_melted = df.melt(value_vars=['log_alpha_i', 'log_alpha_not_i'], 
                        var_name='Parameter', value_name='Log_Value')
    
    df_melted['Parameter'] = df_melted['Parameter'].replace({
        'log_alpha_i': r'$\alpha_i$',
        'log_alpha_not_i': r'$\alpha_{\neg i}$'
    })

    print("Generating log-distribution plot...")
    
    p = (
        ggplot(df_melted, aes(x='Log_Value', fill='Parameter'))
        + geom_histogram(bins=80, alpha=0.90, position='identity', show_legend=True)
        + labs(
            x='Parameter value',
            y='Frequency',
            title=r'Empirical distributions of $\alpha_i$ and $\alpha_{\neg i}$',
            fill='Parameter'
        )
        + scale_fill_manual(values={
            r'$\alpha_i$': '#619CFF', 
            r'$\alpha_{\neg i}$': '#FDB863'
        })
        + theme_minimal()
    )

    p.save("parameter-distribution-20ng.pdf", width=10, height=6, dpi=300)
    print("Plot saved successfully.")

if __name__ == "__main__":
    main()