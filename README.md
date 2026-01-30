# Common TF–IDF variants arise as key components in the test statistic of a penalized likelihood-ratio test for word burstiness

This repository contains computer code for reproducing the results numerical described in the manuscript “Common TF–IDF variants arise as key components in the test statistic of a penalized likelihood-ratio test for word burstiness”

## Getting Started

Clone this repository by running the command
```
git clone https://github.com/sheridan-stable/tfidf-lrt.git
```
and `cd` into the repository root folder `tfidf-lrt`.
```
cd tfidf-lrt
```

## Reproducing Plots and Tables

This section steps through how to reproduce the results described in the manuscript.

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

### Figures

**Figure 1**
```bash
python3 bcb-precision-plot.py
```

**Figure 2**
```bash
python3 bcb-comparison-plots.py
```

**Figure 3**
```bash
cd simulation
python3 tfidf-lambda-plot.py
```

**Figure 4**
```bash
cd twenty-newsgroups/params-verification
python3 mle-on-20ng.py
python3 plot-params.py
```

**Figure 5**
```bash
cd r8/params-verification
python3 mle-on-r8.py
python3 plot-params.py
```

**Figure A1**
```bash
cd twenty-newsgroups/sensitivity-analysis
python3 plot-mu-sigma-grid.py
```

**Figure A2**
```bash
cd r8/sensitivity-analysis
python3 plot-mu-sigma-grid.py
```

### Tables

**Table 2**
```bash
cd twenty-newsgroups/text-classification
python3 twenty-newsgroups.py
```

**Table 3**
```bash
cd r8/text-classification
python3 r8.py
```

## Outputs

The scripts generate the following outputs corresponding to the manuscript figures and tables:

- **Figure 1**: `bcb_precision_plot.pdf`
- **Figure 2**: `bcb_comparison_plot.pdf`
- **Figure 3**: `simulation/tfidf-lambda.pdf` (and `simulation/reports/tfidf-lambda-correlation.txt`)
- **Figure 4**: `twenty-newsgroups/plots/20ng-params-mle.pdf` (from `plot-params.py`, data from `reports/20ng-bb-params-mle.csv`)
- **Figure 5**: `r8/plots/r8-params-mle.pdf` (from `plot-params.py`, data from `reports/r8-bb-params-mle.csv`)
- **Figure A1**: `twenty-newsgroups/plots/20ng-sensitivity-analysis-grid.pdf`
- **Figure A2**: `r8/plots/r8-sensitivity-analysis-grid.pdf`
- **Table 2**: `twenty-newsgroups/reports/tf-idf-report.txt` and `twenty-newsgroups/reports/lambda-i-report.txt`
- **Table 3**: `r8/reports/r8-tfidf-report.txt` and `r8/reports/r8-lambda-i-report.txt`



## Questions and Feedback
If you have a technical question about the manuscript, feel free to post it as an [issue](https://github.com/Sheridan-Stable/tfidf-lrt/issues).

For more open-ended inquiries, we encourage starting a [discussion](https://github.com/Sheridan-Stable/tfidf-lrt/discussions).


## Citation
If you find anything useful please cite our work using:
```
@misc{Ahmed2025,
  author = {Zeyad Ahmed and Paul Sheridan and Michael McIsaac and Aitazaz A. Farooque},
  title = {Common {TF}–{IDF} variants arise as key components in the test statistic of a penalized likelihood-ratio test for word burstiness},
  year = {2025},
  eprint = {arXiv:XXXX.XXXXX}
}
```
