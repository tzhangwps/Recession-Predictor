"""
This module plots results for the deployed model.
"""

import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import matplotlib.pyplot as plt

import RecessionPredictor_paths as path


class TestResultPlots:
    """
    The manager class for this module.
    """
    
    
    def __init__(self):
        """
        prediction_names: used to re-label chart data
        """
        self.pdf_object = ''
        self.prediction_names = {'Pred_Recession_within_6mo': 'Within 6 Months',
                                 'Pred_Recession_within_12mo': 'Within 12 Months',
                                 'Pred_Recession_within_24mo': 'Within 24 Months',
                                 'True_Recession': 'Recession'}
    
    
    def exponential_smoother(self, raw_data, half_life):
        """
        Purpose: performs exponential smoothing on "raw_data". Begins recursion
        with the first data item (i.e. assumes that data in "raw_data" is listed
        in chronological order).
        
        Output: a list containing the smoothed values of "raw_data".
        
        "raw_data": iterable, the data to be smoothed.
        
        "half_life": float, the half-life for the smoother. The smoothing factor
        (alpha) is calculated as alpha = 1 - exp(ln(0.5) / half_life)
        """
        import math
        
        raw_data = list(raw_data)
        half_life = float(half_life)
        
        smoothing_factor = 1 - math.exp(math.log(0.5) / half_life)
        smoothed_values = [raw_data[0]]
        for index in range(1, len(raw_data)):
            previous_smooth_value = smoothed_values[-1]
            new_unsmooth_value = raw_data[index]
            new_smooth_value = ((smoothing_factor * new_unsmooth_value)
                + ((1 - smoothing_factor) * previous_smooth_value))
            smoothed_values.append(new_smooth_value)
        
        return(smoothed_values)
    
    
    def exponential_conversion(self, dataframe):
        """
        Performs exponential smoothing on specific columns of the dataframe.
        """
        dataframe['Within 6 Months'] = self.exponential_smoother(raw_data=dataframe['Within 6 Months'],
                                                                 half_life=3)
        dataframe['Within 12 Months'] = self.exponential_smoother(raw_data=dataframe['Within 12 Months'],
                                                                  half_life=3)
        dataframe['Within 24 Months'] = self.exponential_smoother(raw_data=dataframe['Within 24 Months'],
                                                                  half_life=3)
        return(dataframe)
    
    
    def plot_probabilities(self, dataframe, name, exponential):
        """
        Sets chart parameters, generates the chart, and saves it.
        
        dataframe: dataframe, the dataframe to be plotted
        
        name: sting, chart title
        
        exponential: boolean, whether to plot exponentially weighted output data or not
        """
        dataframe.rename(columns=self.prediction_names, inplace=True)
        if exponential == True:
            dataframe = self.exponential_conversion(dataframe=dataframe)
        is_recession = dataframe['Recession'] == 1
        is_not_recession = dataframe['Recession'] == 0
        dataframe.loc[is_recession, 'Recession'] = 100
        dataframe.loc[is_not_recession, 'Recession'] = -1
        dataframe = dataframe[['Dates'] + list(self.prediction_names.values())]
        
        plt.figure(figsize=(15, 5))
        plot = sns.lineplot(x='Dates', y='value', hue='variable',
                            data=pd.melt(dataframe, ['Dates']))
        plot.set_ylabel('Probability')
        plot.set_title(name, fontsize = 20)
        plot.set_ylim((0, 1))
        self.pdf_object.savefig()
            
    
    def create_chart_data(self, dataframe):
        """
        Reformats data so that it can be uploaded to Visualizer (Wordpress library).
        """
        dataframe.rename(columns=self.prediction_names, inplace=True)
        dataframe = self.exponential_conversion(dataframe=dataframe)
        is_recession = dataframe['Recession'] == 1
        is_not_recession = dataframe['Recession'] == 0
        dataframe.loc[is_recession, 'Recession'] = 100
        dataframe.loc[is_not_recession, 'Recession'] = -1
        dataframe = dataframe[['Dates'] + list(self.prediction_names.values())]
        
        chart_data = pd.DataFrame({
                                   'Dates': ['date'] + list(dataframe['Dates']),
                                   'Within 6 Months': ['number'] + list(dataframe['Within 6 Months']),
                                   'Within 12 Months': ['number'] + list(dataframe['Within 12 Months']),
                                   'Within 24 Months': ['number'] + list(dataframe['Within 24 Months']),
                                   'Recession': ['number'] + list(dataframe['Recession'])
                                   })
        chart_data.to_csv(path.deployment_chart_data, index=False)
    
    
    def plot_test_results(self):
        """
        Loads test results for the deployed model, and plots it into a single PDF.
        """        
        self.svm_test_results = pd.read_json(path.deployment_svm_test_results)
        self.svm_test_results.sort_index(inplace=True)
        print('\nPlotting test results...')
        self.pdf_object = PdfPages(path.deployment_results_plots)
        print('\t|--Plotting SVM test results...')
        self.plot_probabilities(dataframe=self.svm_test_results,
                                name='SVM', exponential=False)
        self.svm_test_results = pd.read_json(path.deployment_svm_test_results)
        self.svm_test_results.sort_index(inplace=True)
        self.plot_probabilities(dataframe=self.svm_test_results,
                                name='SVM EMA', exponential=True)  
        print('\nPlotted results saved to {}'.format(path.deployment_results_plots))
        self.svm_test_results = pd.read_json(path.deployment_svm_test_results)
        self.svm_test_results.sort_index(inplace=True)  
        self.create_chart_data(dataframe=self.svm_test_results)
        print('\nChart data saved to {}'.format(path.deployment_chart_data))
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