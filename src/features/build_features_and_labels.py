"""
This module builds some additional features, labels the output, and consolidates
features and output into the final dataset.
"""
import pandas as pd

import RecessionPredictor_paths as path

class FinalizeDataset:
    """
    The manager class for this module.
    """
    
    
    def __init(self):
        self.secondary_df_output = pd.DataFrame()
        self.final_df_output = pd.DataFrame()

        
    def make_features(self):
        """
        Creates additional features, and adds them to the final dataframe.
        """
        payrolls_3mo_vs_12mo = (self.secondary_df_output['Payrolls_3mo_pct_chg_annualized']
            - self.secondary_df_output['Payrolls_12mo_pct_chg'])
        CPI_3mo_vs_12mo = (self.secondary_df_output['CPI_3mo_pct_chg_annualized']
            - self.secondary_df_output['CPI_12mo_pct_chg'])
        sp500_3mo_vs_12mo = (self.secondary_df_output['S&P_500_3mo_chg']
            - self.secondary_df_output['S&P_500_12mo_chg'])
        IPI_3mo_vs_12mo = (self.secondary_df_output['IPI_3mo_pct_chg_annualized']
            - self.secondary_df_output['IPI_12mo_pct_chg'])
        
        self.final_df_output['Payrolls_3mo_vs_12mo'] = payrolls_3mo_vs_12mo
        self.final_df_output['CPI_3mo_vs_12mo'] = CPI_3mo_vs_12mo
        self.final_df_output['S&P_500_3mo_vs_12mo'] = sp500_3mo_vs_12mo
        self.final_df_output['IPI_3mo_vs_12mo'] = IPI_3mo_vs_12mo
        
        
    def label_output(self):
        """
        Labels the various outputs.
        """
        NBER_recessions = {'1': {'Begin': '1957-08-01', 'End': '1958-04-01'},
                           '2': {'Begin': '1960-04-01', 'End': '1961-02-01'},
                           '3': {'Begin': '1969-12-01', 'End': '1970-11-01'},
                           '4': {'Begin': '1973-11-01', 'End': '1975-03-01'},
                           '5': {'Begin': '1980-01-01', 'End': '1980-07-01'},
                           '6': {'Begin': '1981-07-01', 'End': '1982-11-01'},
                           '7': {'Begin': '1990-07-01', 'End': '1991-03-01'},
                           '8': {'Begin': '2001-03-01', 'End': '2001-11-01'},
                           '9': {'Begin': '2007-12-01', 'End': '2009-06-01'}}
        
        observation_count = len(self.final_df_output)
        self.final_df_output['Recession'] = [0] * observation_count
        self.final_df_output['Recession_in_6mo'] = [0] * observation_count
        self.final_df_output['Recession_in_12mo'] = [0] * observation_count
        self.final_df_output['Recession_in_24mo'] = [0] * observation_count
        self.final_df_output['Recession_within_6mo'] = [0] * observation_count
        self.final_df_output['Recession_within_12mo'] = [0] * observation_count
        self.final_df_output['Recession_within_24mo'] = [0] * observation_count
        
        for recession in NBER_recessions:
            end_condition = (NBER_recessions[recession]['End']
                >= self.final_df_output['Dates'])
            begin_condition = (self.final_df_output['Dates']
                >= NBER_recessions[recession]['Begin'])
            self.final_df_output.loc[end_condition & begin_condition, 'Recession'] = 1
        
        for index in range(0, len(self.final_df_output)):
            if self.final_df_output['Recession'][index] == 1:
                self.final_df_output.loc[min(index + 6, len(self.final_df_output) - 1),
                                         'Recession_in_6mo'] = 1
                self.final_df_output.loc[min(index + 12, len(self.final_df_output) - 1),
                                         'Recession_in_12mo'] = 1
                self.final_df_output.loc[min(index + 24, len(self.final_df_output) - 1),
                                         'Recession_in_24mo'] = 1
                self.final_df_output.loc[index : min(index + 6, len(self.final_df_output) - 1),
                                         'Recession_within_6mo'] = 1
                self.final_df_output.loc[index : min(index + 12, len(self.final_df_output) - 1),
                                         'Recession_within_12mo'] = 1
                self.final_df_output.loc[index : min(index + 24, len(self.final_df_output) - 1),
                                         'Recession_within_24mo'] = 1
        
        
    def create_final_dataset(self):
        """
        Creates and saves the final dataset.
        """
        print('\nCreating final dataset...')
        self.secondary_df_output = pd.read_json(path.data_secondary_most_recent)
        self.secondary_df_output.sort_index(inplace=True)        
        self.final_df_output = self.secondary_df_output
        self.make_features()
        self.label_output()
        new_cols = ['Dates', 'Recession', 'Recession_in_6mo',
                    'Recession_in_12mo', 'Recession_in_24mo',
                    'Recession_within_6mo', 'Recession_within_12mo',
                    'Recession_within_24mo', 'Payrolls_3mo_pct_chg_annualized',
                    'Payrolls_12mo_pct_chg', 'Payrolls_3mo_vs_12mo',
                    'Unemployment_Rate', 'Unemployment_Rate_12mo_chg',
                    'Real_Fed_Funds_Rate', 'Real_Fed_Funds_Rate_12mo_chg',
                    'CPI_3mo_pct_chg_annualized', 'CPI_12mo_pct_chg',
                    'CPI_3mo_vs_12mo', '10Y_Treasury_Rate_12mo_chg',
                    '3M_Treasury_Rate_12mo_chg', '3M_10Y_Treasury_Spread',
                    '3M_10Y_Treasury_Spread_12mo_chg',
                    '5Y_10Y_Treasury_Spread', 'S&P_500_3mo_chg',
                    'S&P_500_12mo_chg', 'S&P_500_3mo_vs_12mo',
                    'IPI_3mo_pct_chg_annualized', 'IPI_12mo_pct_chg',
                    'IPI_3mo_vs_12mo']
        self.final_df_output = self.final_df_output[new_cols]
        print('Finished creating final dataset!')
        print('\t|--Saving final dataset to {}'.format(path.data_final))
        self.final_df_output.to_json(path.data_final)
        print('\nFinal dataset saved to {}'.format(path.data_final))
        
        
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