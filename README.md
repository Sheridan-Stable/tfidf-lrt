# Common TF–IDF variants arise as leading terms in a penalized likelihood-ratio test for word burstiness

This repository contains computer code for reproducing the results numerical described in the manuscript “Common TF–IDF variants arise as leading terms in a penalized likelihood-ratio test for word burstiness.”

## Getting Started

Clone this repository by running the command
```
git clone https://github.com/sheridan-stable/tfidf-lrt.git
```
and `cd` into the repository root folder `tfidf-lrt`.

## Reproducing Plots and Tables
This section steps through how to reproduce the results from Figures 1, 2, 3, and Table 2 in the manuscript. Repository code is written in Python 3. Below is one way to reproduce each plot:

From the command line, create a virtual environment:
```
python3 -m venv .
source bin/activate
```
Install required libraries:
```
pip install -r requirements.txt
```

Each Python script can be run independently:

```bash
# Generate beta-binomial plots (Figure 1)
python3 bcb-precision-plot.py

# Generate likelihood surface comparison plots (Figure 2)
python3 bcb-comparison-plots.py

# Generate lambda_i vs TF-IDF scatter plot (Figure 3)
python3 tfidf-lambda-plot.py

# Generate text classification results (Table 2)
python3 twenty-newsgroups.py
```

## Outputs

The scripts generate several visualization outputs:

- `bcb_precision_plot.pdf`: Visualization of beta-binomial distributions under different parameters (Figure 1)
- `bcb_comparison_plot.pdf`: Contour and 3D surface plots comparing penalized and unpenalized beta-binomial log-likelihood functions (Figure 2)
- `lambda_tfidf.pdf`: Scatter plot showing relationship between lambda_i scores and TF-IDF scores (Figure 3)
- `tf-idf-report.txt`: Classification performance metrics using TF-IDF (Table 2)
- `lambda-i-report.txt`: Classification performance metrics using $S(\lambda_i)$ (Table 2)



## Questions and Feedback
If you have a technical question about the manuscript, feel free to post it as an [issue](https://github.com/Sheridan-Stable/tfidf-lrt/issues).

For more open-ended inquiries, we encourage starting a [discussion](https://github.com/Sheridan-Stable/tfidf-lrt/discussions).


## Citation
If you find anything useful please cite our work using:
```
@misc{Ahmed2025,
  author = {Zeyad Ahmed and Paul Sheridan and Michael McIsaac and Aitazaz A. Farooque},
  title = {Common {TF}–{IDF} variants arise as leading terms in a penalized likelihood-ratio test for word burstiness},
  year = {2025},
  eprint = {arXiv:XXXX.XXXXX}
}
```
