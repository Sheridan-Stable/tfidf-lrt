import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng
from mpl_toolkits.mplot3d import Axes3D

def simulate_corpus(d, n_j, alpha_i_true, alpha_not_i_true, rng):
    """Simulate corpus data given true alpha parameters."""
    theta = rng.dirichlet([alpha_i_true, alpha_not_i_true], size=d)
    counts = rng.multinomial(n_j, pvals=theta)
    n_ij = counts[:, 0]
    n_not_ij = counts[:, 1]
    bij = (n_ij > 0).astype(float)
    return n_not_ij, bij


def compute_loglike_surfaces(A_i, A_not_i, n_not_ij, bij, d, n_j, a, b):
    """Compute unpenalized and penalized approximate log-likelihood surfaces."""
    loglike_approx = np.zeros_like(A_i)
    loglike_approx_penalized = np.zeros_like(A_i)

    for i in range(A_i.shape[0]):
        for j in range(A_i.shape[1]):
            alpha_i = A_i[i, j]
            alpha_not_i = A_not_i[i, j]
            alpha_0 = alpha_i + alpha_not_i

            gamma_penalty = (a - 1) * np.log(alpha_0) - b * alpha_0
            ll_approx = (
                np.sum(bij * np.log(alpha_i))
                + np.sum(n_not_ij * np.log(alpha_not_i))
                - d * n_j * np.log(alpha_0)
            )

            loglike_approx[i, j] = ll_approx
            loglike_approx_penalized[i, j] = ll_approx + gamma_penalty

    return loglike_approx, loglike_approx_penalized


def plot_surfaces(A_i, A_not_i, loglike_approx, loglike_approx_penalized):
    """Create contour and surface plots for the log-likelihoods of Figure 2 in the manuscript."""
    fig = plt.figure(figsize=(10, 7))
    cmap, levels = "plasma", 30

    fig.text(0.5, 0.95, "Approximate beta-binomial model log-likelihood",
             ha="center", va="center", fontsize=12)
    fig.text(0.5, 0.48, "Approximate penalized beta-binomial model log-likelihood",
             ha="center", va="center", fontsize=12)

    # Unpenalized contour
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.contourf(A_not_i, A_i, loglike_approx, levels=levels, cmap=cmap)
    ax1.set_xlabel(r"$\alpha_{\neg i}$")
    ax1.set_ylabel(r"$\alpha_i$")

    # Unpenalized surface
    ax2 = fig.add_subplot(2, 2, 2, projection="3d")
    ax2.plot_surface(A_not_i, A_i, loglike_approx, cmap=cmap, linewidth=0, alpha=0.9)
    ax2.set_xlabel(r"$\alpha_{\neg i}$")
    ax2.set_ylabel(r"$\alpha_i$")
    ax2.set_zlabel("log-likelihood")

    # Penalized contour
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.contourf(A_not_i, A_i, loglike_approx_penalized, levels=levels, cmap=cmap)
    ax3.set_xlabel(r"$\alpha_{\neg i}$")
    ax3.set_ylabel(r"$\alpha_i$")

    # Penalized surface
    ax4 = fig.add_subplot(2, 2, 4, projection="3d")
    ax4.plot_surface(A_not_i, A_i, loglike_approx_penalized,
                     cmap=cmap, linewidth=0, alpha=0.9)
    ax4.set_xlabel(r"$\alpha_{\neg i}$")
    ax4.set_ylabel(r"$\alpha_i$")
    ax4.set_zlabel("log-likelihood")

    # Transparent 3D panes
    for ax in [ax2, ax4]:
        ax.set_facecolor((0, 0, 0, 0))
        ax.xaxis.set_pane_color((1, 1, 1, 0))
        ax.yaxis.set_pane_color((1, 1, 1, 0))
        ax.zaxis.set_pane_color((1, 1, 1, 0))
        ax.grid(False)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.subplots_adjust(hspace=0.4)
    return fig


def find_mle(A_i, A_not_i, loglike):
    """Find MLEs for a likelihood surface."""
    idx_pen = np.unravel_index(np.argmax(loglike), loglike.shape)
    return (A_i[idx_pen], A_not_i[idx_pen])


def main():
    rng = default_rng(0)
    d, n_j = 40, 50
    alpha_i_true, alpha_not_i_true = 0.1, 49.9

    n_not_ij, bij = simulate_corpus(d, n_j, alpha_i_true, alpha_not_i_true, rng)

    # Grid setup
    alpha_i_vals = np.logspace(-10, 0, 100)
    alpha_not_i_vals = np.logspace(1, 2, 100)
    A_i, A_not_i = np.meshgrid(alpha_i_vals, alpha_not_i_vals)

    # Gamma penalty
    mu, sigma = n_j, 2
    a = (mu**2) / (sigma**2)
    b = mu / (sigma**2)

    # Compute likelihoods
    loglike_approx, loglike_approx_penalized = compute_loglike_surfaces(
        A_i, A_not_i, n_not_ij, bij, d, n_j, a, b
    )

    # Plot
    print("\nGenerating interactive 3D plot of Figure 2 ...")
    plot = plot_surfaces(A_i, A_not_i, loglike_approx, loglike_approx_penalized)
    plt.show()
    plot.savefig("bcb_comparison_plot.pdf", transparent=True)
    print("\nPlot saved as bcb_comparison_plot.pdf")


    # MLEs
    print("\nCalculating penalized MLEs...")
    mle_results = find_mle(A_i, A_not_i, loglike_approx_penalized)
    print("Penalized MLEs:")
    print(f"  alpha_i     = {mle_results[0]}")
    print(f"  alpha_not_i = {mle_results[1]}")


if __name__ == "__main__":
    main()
