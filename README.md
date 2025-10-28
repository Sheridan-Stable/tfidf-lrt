# Common TF–IDF variants arise as leading terms in a penalized likelihood-ratio test for word burstiness

This repository contains computer code for reproducing the results numerical described in the manuscript “Common TF–IDF variants arise as leading terms in a penalized likelihood-ratio test for word burstiness.”

## Getting Started

Clone this repository by running the command
```
git clone https://github.com/sheridan-stable/tfidf-lrt.git
```
and `cd` into the repository root folder `tfidf-lrt`.

## Reproducing Plots
This section steps through how to reproduce the results from Figures 1, 2, and 3 in the manuscript. Repository code is written in Python 3. Below is one way to reproduce each plot:

From the command line, create a virtual environment:

## Requirements

- Python 3.x
- NumPy
- Pandas
- Scikit-learn
- Plotnine
- Matplotlib
- SciPy

## Usage

Each Python script can be run independently:

```bash
# Text classification comparison
python twenty-newsgroups.py

# Generate beta-binomial precision plots
python bcb-precision-plot.py

# Create likelihood surface comparison plots
python bcb-comparison-plots.py

# Generate lambda vs TF-IDF scatter plots
python tfidf-lambda-plot.py
```

## Outputs

The scripts generate several visualization outputs:

- `bcb_precision_plot.pdf`: Visualization of beta-binomial distributions under different α parameters
- `bcb_comparison_plot.pdf`: 3D surface plots comparing penalized and unpenalized log-likelihood functions
- `lambda_tfidf.pdf`: Scatter plot showing relationship between λ scores and TF-IDF scores
- `tf-idf-report.txt`: Classification performance metrics using canonical TF-IDF
- `lambda-i-report.txt`: Classification performance metrics using λ-transformation

## License

See the [LICENSE](LICENSE) file for details.
