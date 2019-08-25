# Recession-Predictor (0.1.0)
## Intro
- What does this code do?: TBU
- [Chart](https://terrencez.com/recession-predictor-chart/)
- [Link to Code](https://github.com/tzhangwps/Recession-Predictor/blob/master/RecessionPredictor_master.py)

## Credits
Author: Terrence Zhang
- [Email me](https://terrencez.com/get-in-touch/)
- [LinkedIn](https://www.linkedin.com/in/terrencezhang/)
- [My blog](https://medium.com/@tzhangwps)
- [Facebook](https://www.facebook.com/terrence.zhang.39)

## For Users
### Dependencies

### How to Use the Code
## For Users
### Dependencies
Python (3.6.4)
\
Modules: [requirements.txt](https://github.com/tzhangwps/Recession-Predictor/blob/master/requirements.txt)

### How to Use the Code
1. Download all folders and files in the repository. Maintain the file organization structure.
2. Get your personal FRED API key [here](https://research.stlouisfed.org/docs/api/api_key.html)
3. Copy and paste your FRED API key into the `fred_api_key` object in Line 9 of `RecessionPredictor_paths.py` on your local computer.
4. Run `RecessionPredictor_mater.py` via the command line. It takes one positional argument `process`, whose choices are `backtest` or `deploy`:
- `backtest`: runs all modules required for backtesting models. These modules get the data, perform exploratory analysis, build features, conduct backtests, and plot results from the backtest.
- `deploy`: runs all modules required for model deployment. These modules get the data, build features, and deploy the chosen model onto the most recent data. Model outputs are saved to th `deployment_chart.csv` file.

## For Developers
License: MIT
\
For a more in depth explanation of the code, [click here](https://github.com/tzhangwps/Recession-Predictor/blob/master/DeveloperGuide.md).
