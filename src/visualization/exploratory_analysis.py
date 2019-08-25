"""
This module performs exploratory analysis by generating visuals / charts and
saving them to PDF.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import RecessionPredictor_paths as path

class ExploratoryAnalysis:
    """
    The manager class for this module.
    """
    
    
    def __init__(self):
        """
        end_date: last date to include in the exploratory dataset
        
        full_feature_columns: names of the features to include in exploratory
        analysis
        
        output_series: names of the outputs to include in exploratory analysis
        """
        self.end_date = '1976-01-01'
        self.exploratory_df = pd.DataFrame()
        self.final_df_output = pd.DataFrame()
        self.pdf_object = ''
        self.full_feature_columns = ['Payrolls_3mo_pct_chg_annualized',
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
        self.output_series = ['Recession', 'Recession_within_6mo',
                              'Recession_within_12mo', 'Recession_within_24mo']
    
    
    def plot_positive_class_counts(self):
        """
        Creates charts that plot the number of positive class instances
        (as % of total class instances).
        """
        
        for series_name in self.output_series:
            plt.figure(figsize=(8, 6))
            chart_title = 'Class Counts: "{}"'.format(series_name)
            plot = sns.barplot(x=series_name, y=series_name,
                               data=self.exploratory_df,
                               estimator=lambda x: len(x) / len(self.exploratory_df) * 100)
            plot.set_ylabel('Percent (%)', fontsize=20)
            plot.set_xlabel('')
            plot.set_ylim([0, 100])
            plot.set_title(chart_title, fontsize = 25)
            plot.tick_params(labelsize=15)
            self.pdf_object.savefig()
        
    
    def plot_feature_output_correlations(self):
        """
        Plots feature-output correlations for each feature and output pair.
        """
        from scipy import stats
        
        for output_name in self.output_series:
            correlations = pd.DataFrame()
            for feature_name in self.full_feature_columns:
                pearson_corr = stats.pearsonr(self.exploratory_df[feature_name],
                                              self.exploratory_df[output_name])
                correlations[feature_name] = [pearson_corr[0]]
            
            plt.figure(figsize=(50, 10))
            chart_title = 'Feature Correlations to "{}" Label'.format(output_name)
            plot = sns.barplot(data=correlations, orient='h')
            plot.set_ylabel('')
            plot.set_xlabel('Correlation', fontsize=30)
            plot.set_title(chart_title, fontsize = 40)
            plot.tick_params(labelsize=20)
            plot.set_xlim((-0.7, 0.7))
            self.pdf_object.savefig()
    
    
    def plot_feature_output_scatterplots(self):
        """
        Plots feature-output scatterplots for each feature and output pair.
        """
        for output_name in self.output_series:
            for feature_name in self.full_feature_columns:
                
                plt.figure(figsize=(8, 6))
                chart_title = '{} vs. {}'.format(output_name, feature_name)
                plot = sns.scatterplot(x=feature_name, y=output_name,
                                       data=self.exploratory_df)
                plot.set_ylim([-0.1, 1.1])
                plot.set_title(chart_title, fontsize = 20)
                plot.tick_params(labelsize=15)
                self.pdf_object.savefig()
    
    
    def plot_correlation_heatmaps(self):
        """
        Plot correlation heatmaps between select individual features.
        """        
        feature_set_1 = ['Payrolls_3mo_vs_12mo', 'Real_Fed_Funds_Rate_12mo_chg',
                         'CPI_3mo_pct_chg_annualized', 'CPI_12mo_pct_chg']
        correlation_matrix = self.exploratory_df[feature_set_1].corr()
        plt.figure(figsize=(5, 5))
        chart_title = 'Correlation Heatmap - Pick Inflation Feature'
        plot = sns.heatmap(correlation_matrix, vmin=-1, vmax=1,
                           xticklabels=correlation_matrix.columns,
                           yticklabels=correlation_matrix.columns,
                           annot=True, fmt='.0%')
        plot.set_title(chart_title, fontsize = 10)
        plot.tick_params(labelsize=4)
        self.pdf_object.savefig()
        
        feature_set_2 = ['Payrolls_3mo_vs_12mo', 'Real_Fed_Funds_Rate_12mo_chg',
                         'CPI_3mo_pct_chg_annualized', 
                         '10Y_Treasury_Rate_12mo_chg',
                         '3M_Treasury_Rate_12mo_chg',
                         '3M_10Y_Treasury_Spread',
                         '3M_10Y_Treasury_Spread_12mo_chg',
                         '5Y_10Y_Treasury_Spread']
        correlation_matrix = self.exploratory_df[feature_set_2].corr()
        plt.figure(figsize=(5, 5))
        chart_title = 'Correlation Heatmap - Pick Treasury Rate Features'
        plot = sns.heatmap(correlation_matrix, vmin=-1, vmax=1,
                           xticklabels=correlation_matrix.columns,
                           yticklabels=correlation_matrix.columns,
                           annot=True, fmt='.0%')
        plot.set_title(chart_title, fontsize = 10)
        plot.tick_params(labelsize=4)
        self.pdf_object.savefig()
        
        feature_set_3 = ['Payrolls_3mo_vs_12mo', 'Real_Fed_Funds_Rate_12mo_chg',
                         'CPI_3mo_pct_chg_annualized', 
                         '10Y_Treasury_Rate_12mo_chg',
                         '3M_10Y_Treasury_Spread', 'S&P_500_12mo_chg',
                         'S&P_500_3mo_vs_12mo']
        correlation_matrix = self.exploratory_df[feature_set_3].corr()
        plt.figure(figsize=(5, 5))
        chart_title = 'Correlation Heatmap - Pick Stock Market Feature'
        plot = sns.heatmap(correlation_matrix, vmin=-1, vmax=1,
                           xticklabels=correlation_matrix.columns,
                           yticklabels=correlation_matrix.columns,
                           annot=True, fmt='.0%')
        plot.set_title(chart_title, fontsize = 10)
        plot.tick_params(labelsize=4)
        self.pdf_object.savefig()
        
        feature_set_4 = ['Payrolls_3mo_vs_12mo', 'Real_Fed_Funds_Rate_12mo_chg',
                         'CPI_3mo_pct_chg_annualized', 
                         '10Y_Treasury_Rate_12mo_chg',
                         '3M_10Y_Treasury_Spread', 'S&P_500_12mo_chg']
        pd.plotting.scatter_matrix(self.exploratory_df[feature_set_4],
                                  figsize=(18, 18), diagonal='kde')
        self.pdf_object.savefig()
        
        
    def explore_dataset(self):
        """
        Performs exploratory analysis on the final dataset.
        """
        self.final_df_output = pd.read_json(path.data_final)
        self.final_df_output.sort_index(inplace=True)
        end_date_condition = self.final_df_output['Dates'] <= self.end_date
        self.exploratory_df = self.final_df_output[end_date_condition]
        
        print('\nCreating exploratory plots...')
        self.pdf_object = PdfPages(path.exploratory_plots)
        print('\t|--Plotting "Positive Class" instances, for each Output Type...')
        self.plot_positive_class_counts()
        print('\t|--Plotting pairwise correlations, for each Output Type, by each Feature...')
        self.plot_feature_output_correlations()
        print('\t|--Plotting pairwise scatterplots, for each Output Type, by each Feature...')
        self.plot_feature_output_scatterplots()
        print('\t|--Plotting feature correlation heatmaps...')
        self.plot_correlation_heatmaps()        
        print('Plots completed and saved to {}'.format(path.exploratory_plots))
        self.pdf_object.close()
        
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