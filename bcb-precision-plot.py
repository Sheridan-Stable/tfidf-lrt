import numpy as np
import pandas as pd
from plotnine import *

# Custom colors
BLUE = "#619CFF"
GRAY = "#404040"

# Constants
D = 500
N_J = 100

PARAMS_LIST = [
    ([0.03, 0.07], r"$\alpha_{0i}=0.1$"+'\n'+r"($\alpha_i=0.03,\ \alpha_{\neg i}=0.07$)"),
    ([0.3, 0.7], r"$\alpha_{0i}=1.0$"+'\n'+r"($\alpha_i=0.3,\ \alpha_{\neg i}=0.7$)"),
    ([3, 7], r"$\alpha_{0i}=10$"+'\n'+r"($\alpha_i=3,\ \alpha_{\neg i}=7$)"),
    ([30, 70], r"$\alpha_{0i}=100$"+'\n'+r"($\alpha_i=30,\ \alpha_{\neg i}=70$)")
]


def generate_data(params_list, num_docs, doc_length):
    """Generate beta-binomial samples and return a combined DataFrame."""
    dfs = []
    for (alpha_i, alpha_not_i), label in params_list:
        thetas = np.random.beta(alpha_i, alpha_not_i, size=num_docs)
        counts = np.random.binomial(n=doc_length, p=thetas)
        dfs.append(pd.DataFrame({'count': counts, 'alpha_label': label}))
    return pd.concat(dfs, ignore_index=True)


def make_plot(df, n, blue, gray):
    """Create the beta-binomial plot of Figure 1 in the manuscript."""
    return (
        ggplot(df, aes(x='count', y=0))
        + geom_jitter(width=0, height=0.1, alpha=0.75, size=2.5, color=blue, stroke=0)
        + geom_hline(yintercept=0, linetype='dashed', color=gray, size=0.5, alpha=0.2)
        + geom_vline(xintercept=30, linetype='dashed', color=gray, size=0.5, alpha=0.2)
        + xlim(0, n)
        + ylim(-0.3, 0.3)
        + facet_wrap('~alpha_label', nrow=2)
        + labs(
            x=r"Term counts ($n_{ij}$)",
            y=None,
            title=r"Effect of $\alpha_{0i}$ on the beta–binomial distribution"
        )
        + theme_minimal()
        + theme(
            plot_title=element_text(margin={'b': 10}),
            axis_title_y=element_blank(),
            axis_text_y=element_blank(),
            axis_title_x=element_text(margin={'t': 10}),
            axis_text_x=element_text(va='top', y=-0.02),
            panel_grid_minor=element_blank(),
            panel_grid_major=element_blank(),
            panel_border=element_rect(color=gray, fill=None, size=0.3),
            figure_size=(8, 4)
        )
    )


def main():
    np.random.seed(0)
    print("Generating beta-binomial plot of Figure 1...")
    df_all = generate_data(PARAMS_LIST, D, N_J)
    plot = make_plot(df_all, N_J, BLUE, GRAY)
    plot.save("bcb_precision_plot.pdf", dpi=300)
    print("Plot saved as bcb_precision_plot.pdf")


if __name__ == "__main__":
    main()
