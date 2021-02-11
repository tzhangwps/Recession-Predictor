"""
This module gets data from FRED and Yahoo Finance, builds some features,
and saves the data into the respective filepaths.
"""

import re
from io import StringIO
import json
from datetime import datetime, timedelta
import requests as req
import pandas as pd

import RecessionPredictor_paths as path


class YahooData:
    """
    Retrieves data from Yahoo Finance.
    
    Original code source: https://stackoverflow.com/questions/44225771/scraping-historical-data-from-yahoo-finance-with-python
    """
    timeout = 2
    crumb_link = 'https://finance.yahoo.com/quote/{0}/history?p={0}'
    crumble_regex = r'CrumbStore":{"crumb":"(.*?)"}'
    quote_link = 'https://query1.finance.yahoo.com/v7/finance/download/{quote}?period1={dfrom}&period2={dto}&interval=1mo&events=history&crumb={crumb}'


    def __init__(self, symbol, days_back=7):
        """
        symbol: ticker symbol for the asset to be pulled.
        """
        self.symbol = str(symbol)
        self.session = req.Session()
        self.dt = timedelta(days=days_back)


    def get_crumb(self):
        """
        Original code source: https://stackoverflow.com/questions/44225771/scraping-historical-data-from-yahoo-finance-with-python
        """
        response = self.session.get(self.crumb_link.format(self.symbol), timeout=self.timeout)
        response.raise_for_status()
        match = re.search(self.crumble_regex, response.text)
        if not match:
            raise ValueError('Could not get crumb from Yahoo Finance')
        else:
            self.crumb = match.group(1)


    def get_quote(self):
        """
        Original code source: https://stackoverflow.com/questions/44225771/scraping-historical-data-from-yahoo-finance-with-python
        """
        if not hasattr(self, 'crumb') or len(self.session.cookies) == 0:
            self.get_crumb()
        now = datetime.utcnow()
        dateto = int(now.timestamp())
        datefrom = -630961200
#       line in original code: datefrom = int((now - self.dt).timestamp())
        url = self.quote_link.format(quote=self.symbol, dfrom=datefrom, dto=dateto, crumb=self.crumb)
        response = self.session.get(url)
        response.raise_for_status()
        return pd.read_csv(StringIO(response.text), parse_dates=['Date'])


class DataSeries:
    """
    Contains methods and objects to retrieve data from FRED and Yahoo Finance.
    """
    
    
    def __init__(self):
        self.dates = []
        self.values = []
        
        
    def fred_response(self, params):
        """
        Makes requests to the FRED API.
        
        params: dictionary, FRED API parameters.
        """
        params = dict(params)
        fred_request = req.get(url='https://api.stlouisfed.org/fred/series/observations',
                               params=params)
        fred_json = json.loads(fred_request.text)['observations']
        for observation in fred_json:
            self.dates.append(str(observation['date']))
            self.values.append(float(observation['value']))
         
            
    def yahoo_response(self, series_id):
        """
        Retrieves data from Yahoo Finance, and performs timestamp adjustments.
        
        series_id: ticker symbol for the asset to be pulled.
        """
        series_id = str(series_id)
        series_dataframe = YahooData(series_id).get_quote()[::-1]
        series_dataframe.reset_index(inplace=True)
        series_dataframe.drop('index', axis=1, inplace=True)
        most_recent_day = datetime.strptime(str(series_dataframe['Date'][0])[:10],
                                            '%Y-%m-%d').day
        if most_recent_day != 1:
            series_dataframe = series_dataframe[1:]
            series_dataframe.reset_index(inplace=True)
            series_dataframe.drop('index', axis=1, inplace=True)
        self.dates.extend([str(series_dataframe['Date'][index])[:10]
            for index in range(0, len(series_dataframe))])
        self.values.extend([float(series_dataframe['Adj Close'][index])
            for index in range(0, len(series_dataframe))])
        
        
class MakeDataset:
    """
    The manager class for this module.
    """

    
    def __init__(self):
        """
        fred_series_ids: identifiers for FRED data series.
        
        yahoo series_ids: identifiers for Yahoo Finance data series.
        """
        self.fred_series_ids = {'Non-farm_Payrolls': 'PAYEMS',
                                'Civilian_Unemployment_Rate': 'UNRATE',
                                'Effective_Fed_Funds': 'FEDFUNDS',
                                'CPI_All_Items': 'CPIAUCSL',
                                '10Y_Treasury_Rate': 'GS10',
                                '5Y_Treasury_Rate': 'GS5',
                                '3_Month_T-Bill_Rate': 'TB3MS',
                                'IPI': 'INDPRO'}
        self.yahoo_series_ids = {'S&P_500_Index': '^GSPC'}
        self.primary_dictionary_output = {}
        self.primary_df_output = pd.DataFrame()
        self.shortest_series_name = ''
        self.shortest_series_length = 1000000
        self.secondary_df_output = pd.DataFrame()

    
    def get_fred_data(self):
        """
        Cycles through "fred_series"ids" to get data from the FRED API.
        """
        import time
        
        now = datetime.now()
        month = now.strftime('%m')
        year = now.year        
        most_recent_date = '{}-{}-08'.format(year, month)
        print('\nGetting data from FRED API as of {}...'.format(most_recent_date))
        
        for series_name in list(self.fred_series_ids.keys()):
            series_data = DataSeries()
            series_id = self.fred_series_ids[series_name]
            print('\t|--Getting data for {}({}).'.format(series_name, series_id))
            params = {'series_id': series_id,
                      'api_key': path.fred_api_key,
                      'file_type': 'json',
                      'sort_order': 'desc',
                      'realtime_start': most_recent_date,
                      'realtime_end': most_recent_date}
            success = False
            while success == False:
                try:
                    series_data.fred_response(params)
                except json.JSONDecodeError:
                    delay = 5
                    print('\t --CONNECTION ERROR--',
                          '\n\t Sleeping for {} seconds.'.format(delay))
                    time.sleep(delay) 
                else:
                    success = True
            self.primary_dictionary_output[series_name] = series_data
        print('Finished getting data from FRED API!')
    
    
    def get_yahoo_data(self):
        """
        Cycles through "yahoo_series"ids" to get data from the Yahoo Finance.
        """        
        import time
        
        print('\nGetting data from Yahoo Finance...')
        for series_name in list(self.yahoo_series_ids.keys()):
            series_data = DataSeries()
            series_id = self.yahoo_series_ids[series_name]
            print('\t|--Getting data for {}({}).'.format(series_name, series_id))
            success = False
            while success == False:
                try:
                    series_data.yahoo_response(series_id)
                except req.HTTPError:
                    delay = 5
                    print('\t --CONNECTION ERROR--',
                          '\n\t Sleeping for {} seconds.'.format(delay))
                    time.sleep(delay)
                else:
                    success = True
            self.primary_dictionary_output[series_name] = series_data
        print('Finished getting data from Yahoo Finance!')
        
        
    def yahoo_data_sp500_fix(self):
        """
        For some reason Yahoo Finance is no longer providing monthly
        S&P 500 data past the cutoff_date. So will need to retrieve all
        S&P 500 data prior to cutoff_date from a previous run of the code.
        """
        sp500_precutoff_data = pd.read_json(path.sp500_precutoff_data)
        sp500_precutoff_data.sort_index(inplace=True)
        cutoff_date = self.primary_dictionary_output['S&P_500_Index'].dates[::-1][0]
        cutoff_date_mask = sp500_precutoff_data.loc[:,'Dates'] < cutoff_date
        self.primary_dictionary_output['S&P_500_Index'].dates.extend(sp500_precutoff_data.loc[cutoff_date_mask, 'Dates'])
        self.primary_dictionary_output['S&P_500_Index'].values.extend(sp500_precutoff_data.loc[cutoff_date_mask, 'S&P_500_Index'])
        

    def find_shortest_series(self):
        """
        Finds the length and name of the shortes series in the primary
        dataset.
        """
        for series_name in self.primary_dictionary_output.keys():
            series_data = self.primary_dictionary_output[series_name]
            if len(series_data.dates) < self.shortest_series_length:
                self.shortest_series_length = len(series_data.dates)
                self.shortest_series_name = series_name
    
    
    def combine_primary_data(self):
        """
        Combines primary data into a single dictionary (such that each series
        is the same length and is time-matched to each other) and saves it
        as a json object.
        """
        print('\nCombining primary dataset...')
        now = datetime.now()
        current_month = int(now.strftime('%m'))
        current_year = now.year        
        
        dates = []
        for months_ago in range(0, self.shortest_series_length):
            if current_month < 10:
                dates.append('{}-0{}-01'.format(current_year, current_month))
            else:
                dates.append('{}-{}-01'.format(current_year, current_month))
            
            if current_month == 1:
                current_month = 12
                current_year -= 1
            else:
                current_month -= 1
            
        self.primary_df_output['Dates'] = dates
        
        for series_name in self.primary_dictionary_output.keys():
            series_data = self.primary_dictionary_output[series_name]
            self.primary_df_output[series_name] = series_data.values[:self.shortest_series_length]
        print('Finished combining primary dataset!')
        print('\t|--Saving primary dataset to {}'.format(path.data_primary))
        self.primary_df_output.to_json(path.data_primary)
        self.primary_df_output.to_json(path.data_primary_most_recent)
        print('\nPrimary dataset saved to {}'.format(path.data_primary_most_recent))


    def get_primary_data(self):
        """
        Gets primary data from FRED API and Yahoo Finance.
        """
        
        print('\nGetting primary data from APIs...')
        self.get_fred_data()
        self.get_yahoo_data()
        self.yahoo_data_sp500_fix()
        self.find_shortest_series()
        self.combine_primary_data()
        
    
    def calculate_secondary_data(self):
        """
        Builds some features from the primary dataset to create a secondary
        dataset.
        """
        dates = []
        payrolls_3mo = []
        payrolls_12mo = []
        unemployment_rate = []
        unemployment_rate_12mo_chg = []
        real_fed_funds = []
        real_fed_funds_12mo = []
        CPI_3mo = []
        CPI_12mo = []
        treasury_10Y_12mo = []
        treasury_3M_12mo = []
        treasury_10Y_3M_spread = []
        treasury_10Y_5Y_spread = []
        treasury_10Y_3M_spread_12mo = []
        sp_500_3mo = []
        sp_500_12mo = []
        IPI_3mo = []
        IPI_12mo = []
        
        for index in range(0, len(self.primary_df_output) - 12):
            dates.append(self.primary_df_output['Dates'][index])
            payrolls_3mo_pct_chg = (self.primary_df_output['Non-farm_Payrolls'][index]
                / self.primary_df_output['Non-farm_Payrolls'][index + 3]) - 1
            payrolls_3mo.append(((1 + payrolls_3mo_pct_chg) ** 4) - 1)
            payrolls_12mo.append((self.primary_df_output['Non-farm_Payrolls'][index]
                / self.primary_df_output['Non-farm_Payrolls'][index + 12]) - 1)
            unemployment_rate.append(self.primary_df_output['Civilian_Unemployment_Rate'][index])
            unemployment_rate_12mo_chg.append((self.primary_df_output['Civilian_Unemployment_Rate'][index])
                - self.primary_df_output['Civilian_Unemployment_Rate'][index + 12])
            CPI_3mo_pct_chg = (self.primary_df_output['CPI_All_Items'][index]
                / self.primary_df_output['CPI_All_Items'][index + 3]) - 1
            CPI_3mo.append(((1 + CPI_3mo_pct_chg) ** 4) - 1)
            CPI_12mo_pct_chg = (self.primary_df_output['CPI_All_Items'][index]
                / self.primary_df_output['CPI_All_Items'][index + 12]) - 1
            CPI_12mo.append(CPI_12mo_pct_chg)
            real_fed_funds.append(self.primary_df_output['Effective_Fed_Funds'][index]
                - (CPI_12mo_pct_chg * 100))
            real_fed_funds_12mo.append(self.primary_df_output['Effective_Fed_Funds'][index]
                - self.primary_df_output['Effective_Fed_Funds'][index + 12])
            treasury_10Y_12mo.append(self.primary_df_output['10Y_Treasury_Rate'][index]
                - self.primary_df_output['10Y_Treasury_Rate'][index + 12])
            treasury_3M_12mo.append(self.primary_df_output['3_Month_T-Bill_Rate'][index]
                - self.primary_df_output['3_Month_T-Bill_Rate'][index + 12])
            treasury_10Y_3M_spread_today = (self.primary_df_output['10Y_Treasury_Rate'][index]
                - self.primary_df_output['3_Month_T-Bill_Rate'][index])
            treasury_10Y_3M_spread.append(treasury_10Y_3M_spread_today)
            treasury_10Y_3M_spread_12mo_ago = (self.primary_df_output['10Y_Treasury_Rate'][index + 12]
                - self.primary_df_output['3_Month_T-Bill_Rate'][index + 12])
            treasury_10Y_3M_spread_12mo.append(treasury_10Y_3M_spread_today
                                               - treasury_10Y_3M_spread_12mo_ago)
            treasury_10Y_5Y_spread_today = (self.primary_df_output['10Y_Treasury_Rate'][index]
                - self.primary_df_output['5Y_Treasury_Rate'][index])
            treasury_10Y_5Y_spread.append(treasury_10Y_5Y_spread_today)
            sp_500_3mo.append((self.primary_df_output['S&P_500_Index'][index]
                / self.primary_df_output['S&P_500_Index'][index + 3]) - 1)
            sp_500_12mo.append((self.primary_df_output['S&P_500_Index'][index]
                / self.primary_df_output['S&P_500_Index'][index +12]) - 1)
            IPI_3mo_pct_chg = (self.primary_df_output['IPI'][index]
                / self.primary_df_output['IPI'][index + 3]) - 1
            IPI_3mo.append(((1 + IPI_3mo_pct_chg) ** 4) - 1)
            IPI_12mo_pct_chg = (self.primary_df_output['IPI'][index]
                / self.primary_df_output['IPI'][index + 12]) - 1
            IPI_12mo.append(IPI_12mo_pct_chg)
            
        self.secondary_df_output = pd.DataFrame({
                'Dates': dates,
                'Payrolls_3mo_pct_chg_annualized': payrolls_3mo,
                'Payrolls_12mo_pct_chg': payrolls_12mo,
                'Unemployment_Rate': unemployment_rate,
                'Unemployment_Rate_12mo_chg': unemployment_rate_12mo_chg,
                'Real_Fed_Funds_Rate': real_fed_funds,
                'Real_Fed_Funds_Rate_12mo_chg': real_fed_funds_12mo,
                'CPI_3mo_pct_chg_annualized': CPI_3mo,
                'CPI_12mo_pct_chg': CPI_12mo,
                '10Y_Treasury_Rate_12mo_chg': treasury_10Y_12mo,
                '3M_Treasury_Rate_12mo_chg': treasury_3M_12mo,
                '3M_10Y_Treasury_Spread': treasury_10Y_3M_spread,
                '3M_10Y_Treasury_Spread_12mo_chg': treasury_10Y_3M_spread_12mo,
                '5Y_10Y_Treasury_Spread': treasury_10Y_5Y_spread,
                'S&P_500_3mo_chg': sp_500_3mo,
                'S&P_500_12mo_chg': sp_500_12mo,
                'IPI_3mo_pct_chg_annualized': IPI_3mo,
                'IPI_12mo_pct_chg': IPI_12mo})
            
            
    def create_secondary_data(self):
        """
        Creates and saves the secondary dataset as a json object.
        """
        print('\nCreating secondary dataset from "primary_dataset_most_recent.json"')
        self.primary_df_output = pd.read_json(path.data_primary_most_recent)
        self.primary_df_output.sort_index(inplace=True)
        self.calculate_secondary_data()
        print('Finished creating secondary dataset!')
        print('\t|--Saving secondary dataset to {}'.format(path.data_secondary))
        self.secondary_df_output.to_json(path.data_secondary)
        self.secondary_df_output.to_json(path.data_secondary_most_recent)
        print('\nSecondary dataset saved to {}'.format(path.data_secondary_most_recent))
        
        
    def get_all_data(self):
        """
        Gets data from primary sources (FRED and Yahoo Finance), then performs
        preliminary manipulations before saving the data.
        """
        self.get_primary_data()
        self.create_secondary_data()


# FRED citations
#U.S. Bureau of Labor Statistics, All Employees: Total Nonfarm Payrolls [PAYEMS], retrieved from FRED, Federal Reserve Bank of St. Louis; https://fred.stlouisfed.org/series/PAYEMS
#U.S. Bureau of Labor Statistics, Civilian Unemployment Rate [UNRATE], retrieved from FRED, Federal Reserve Bank of St. Louis; https://fred.stlouisfed.org/series/UNRATE
#Board of Governors of the Federal Reserve System (US), Effective Federal Funds Rate [FEDFUNDS], retrieved from FRED, Federal Reserve Bank of St. Louis; https://fred.stlouisfed.org/series/FEDFUNDS
#U.S. Bureau of Labor Statistics, Consumer Price Index for All Urban Consumers: All Items [CPIAUCSL], retrieved from FRED, Federal Reserve Bank of St. Louis; https://fred.stlouisfed.org/series/CPIAUCSL
#Board of Governors of the Federal Reserve System (US), 10-Year Treasury Constant Maturity Rate [GS10], retrieved from FRED, Federal Reserve Bank of St. Louis; https://fred.stlouisfed.org/series/GS10
#Board of Governors of the Federal Reserve System (US), 5-Year Treasury Constant Maturity Rate [GS5], retrieved from FRED, Federal Reserve Bank of St. Louis; https://fred.stlouisfed.org/series/GS5
#Board of Governors of the Federal Reserve System (US), 3-Month Treasury Bill: Secondary Market Rate [TB3MS], retrieved from FRED, Federal Reserve Bank of St. Louis; https://fred.stlouisfed.org/series/TB3MS
#Board of Governors of the Federal Reserve System (US), Industrial Production Index [INDPRO], retrieved from FRED, Federal Reserve Bank of St. Louis; https://fred.stlouisfed.org/series/INDPRO
        
#MIT License
#
#Copyright (c) 2019 Terrence Zhang
#
#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.
