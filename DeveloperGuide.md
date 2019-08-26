# Recession Predictor (0.1.0)
## For Developers
### Code Walkthrough
The following text walks through the main functionalities of the modules.
\
For instructions on how to run the code, [click here](https://github.com/tzhangwps/Recession-Predictor/blob/master/README.md).

### `make_dataset.py`
Gets raw data from the FRED API and Yahoo Finance. The code to get data from Yahoo Finance comes from [this StackOverflow post](https://stackoverflow.com/questions/44225771/scraping-historical-data-from-yahoo-finance-with-python). This module also creates most of the features to be used later on in the analysis.

### `build_features_and_labels.py`
Builds some additional features, and organizes all the raw data into the final dataset to be used in the rest of the analysis.

### `exploratory_analysis.py`
Creates charts for exploratory analysis, and saves all charts into the `exploratory.pdf` file.

### `testing.py`
Runs backtests for each model. Model-specific code is stored in the `/models/` folder.

### `test_results.py`
Plots model predictions (probabilities) in line charts, for all models.

### `deployment.py`
Runs the chosen model in real-time. Model-specific code is stored in the `/models/` folder.

### `deployment_results.py`
Plots model predictions (probabilities) in line charts, for the chosen model. Also outputs a `deployment_chart.csv` file containing the chosen model predictions.
