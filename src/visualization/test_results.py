"""
This module plots test results.
"""

import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss

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
        self.average_model = pd.DataFrame()
    
    
    def calculate_log_loss_weights(self, y_true):
        """
        Calculates weight adjustments for class outputs, such that each class
        receives the same weight in log loss calculations.
        
        y_true: an iterable of class outputs to be weighted.
        """
        log_loss_weights = []
        true_output_labels = y_true.unique()
        desired_weight = 1 / len(true_output_labels)
        
        class_weights = {}
        for label in true_output_labels:
            training_frequency = (len(y_true[y_true == label]) / len(y_true))
            multiplier = desired_weight / training_frequency
            class_weights[str(label)] = multiplier
        
        for sample in y_true:
            log_loss_weights.append(class_weights[str(sample)])
        
        return(log_loss_weights)
    
    
    def exponential_smoother(self, raw_data, half_life):
        """
        Purpose: performs exponential smoothing on "raw_data". Begins recursion
        with the first data item (i.e. assumes that data in "raw_data" is listed
        in chronological order).
        
        raw_data: iterable, the data to be smoothed
        
        half_life: float, the half-life for the smoother. The smoothing factor
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
        dataframe = pd.DataFrame(dataframe)
        name = str(name)
        exponential = bool(exponential)
        
        dataframe.rename(columns=self.prediction_names, inplace=True)
        if exponential == True:
            dataframe = self.exponential_conversion(dataframe=dataframe)
        is_recession = dataframe['Recession'] == 1
        is_not_recession = dataframe['Recession'] == 0
        dataframe.loc[is_recession, 'Recession'] = 100
        dataframe.loc[is_not_recession, 'Recession'] = -1
                
        log_loss_weights_6mo = self.calculate_log_loss_weights(y_true=dataframe['True_Recession_within_6mo'])
        log_loss_weights_12mo = self.calculate_log_loss_weights(y_true=dataframe['True_Recession_within_12mo'])
        log_loss_weights_24mo = self.calculate_log_loss_weights(y_true=dataframe['True_Recession_within_24mo'])
        loss_6mo = log_loss(y_true=dataframe['True_Recession_within_6mo'],
                            y_pred=dataframe['Within 6 Months'],
                            sample_weight=log_loss_weights_6mo)
        loss_12mo = log_loss(y_true=dataframe['True_Recession_within_12mo'],
                             y_pred=dataframe['Within 12 Months'],
                             sample_weight=log_loss_weights_12mo)
        loss_24mo = log_loss(y_true=dataframe['True_Recession_within_24mo'],
                             y_pred=dataframe['Within 24 Months'],
                             sample_weight=log_loss_weights_24mo)
        dataframe = dataframe[['Dates'] + list(self.prediction_names.values())]
        
        chart_title = '{} | 6mo: {} | 12mo: {} | 24mo: {}'.format(name,
                       round(loss_6mo, 3), round(loss_12mo, 3),
                       round(loss_24mo, 3))
        plt.figure(figsize=(15, 5))
        plot = sns.lineplot(x='Dates', y='value', hue='variable',
                            data=pd.melt(dataframe, ['Dates']))
        plot.set_ylabel('Probability')
        plot.set_title(chart_title, fontsize = 20)
        plot.set_ylim((0, 1))
        self.pdf_object.savefig()
    
    
    def average_model_outputs(self):
        """
        Creates outputs for a Grand Average model by averaging across all
        model outputs.
        """
        from statistics import mean
        
        self.average_model['Dates'] = self.knn_test_results['Dates']
        self.average_model['Recession'] = self.knn_test_results['True_Recession']
        self.average_model['True_Recession_within_6mo'] = self.knn_test_results['True_Recession_within_6mo']
        self.average_model['True_Recession_within_12mo'] = self.knn_test_results['True_Recession_within_12mo']
        self.average_model['True_Recession_within_24mo'] = self.knn_test_results['True_Recession_within_24mo']
        model_outputs = [self.knn_test_results, self.elastic_net_test_results,
                         self.naive_bayes_test_results, self.svm_test_results,
                         self.gauss_test_results, self.xgboost_test_results]
        average_6mo = []
        average_12mo = []
        average_24mo = []
        for index in range(0, len(self.knn_test_results)):
            outputs_6mo = []
            outputs_12mo = []
            outputs_24mo = []
            for model in model_outputs:
                outputs_6mo.append(model['Pred_Recession_within_6mo'][index])
                outputs_12mo.append(model['Pred_Recession_within_12mo'][index])
                outputs_24mo.append(model['Pred_Recession_within_24mo'][index])
            average_6mo.append(mean(outputs_6mo))
            average_12mo.append(mean(outputs_12mo))
            average_24mo.append(mean(outputs_24mo))
        
        self.average_model['Within 6 Months'] = average_6mo
        self.average_model['Within 12 Months'] = average_12mo
        self.average_model['Within 24 Months'] = average_24mo
            
    
    def plot_test_results(self):
        """
        Loads test results for each model, and plots them all into a single PDF.
        """
        self.knn_test_results = pd.read_json(path.knn_test_results)
        self.knn_test_results.sort_index(inplace=True)
        self.elastic_net_test_results = pd.read_json(path.elastic_net_test_results)
        self.elastic_net_test_results.sort_index(inplace=True)
        self.naive_bayes_test_results = pd.read_json(path.naive_bayes_test_results)
        self.naive_bayes_test_results.sort_index(inplace=True)
        self.svm_test_results = pd.read_json(path.svm_test_results)
        self.svm_test_results.sort_index(inplace=True)
        self.gauss_test_results = pd.read_json(path.gauss_test_results)
        self.gauss_test_results.sort_index(inplace=True)
        self.xgboost_test_results = pd.read_json(path.xgboost_test_results)
        self.xgboost_test_results.sort_index(inplace=True)
        self.weighted_average_test_results = pd.read_json(path.weighted_average_test_results)
        self.weighted_average_test_results.sort_index(inplace=True)
        
        print('\nPlotting test results...')
        self.pdf_object = PdfPages(path.test_results_plots)
        print('\t|--Plotting KNN test results...')
        self.plot_probabilities(dataframe=self.knn_test_results,
                                name='KNN', exponential=False)
        self.plot_probabilities(dataframe=self.knn_test_results,
                                name='KNN EMA', exponential=True)
        print('\t|--Plotting Elastic Net test results...')
        self.plot_probabilities(dataframe=self.elastic_net_test_results,
                                name='Elastic Net', exponential=False)
        self.plot_probabilities(dataframe=self.elastic_net_test_results,
                                name='Elastic Net EMA', exponential=True)
        print('\t|--Plotting Naive Bayes test results...')
        self.plot_probabilities(dataframe=self.naive_bayes_test_results,
                                name='Naive Bayes', exponential=False)
        self.plot_probabilities(dataframe=self.naive_bayes_test_results,
                                name='Naive Bayes EMA', exponential=True)
        print('\t|--Plotting SVM test results...')
        self.plot_probabilities(dataframe=self.svm_test_results,
                                name='SVM', exponential=False)
        self.plot_probabilities(dataframe=self.svm_test_results,
                                name='SVM EMA', exponential=True)
        print('\t|--Plotting Gaussian Process test results...')
        self.plot_probabilities(dataframe=self.gauss_test_results,
                                name='Gaussian Process', exponential=False)
        self.plot_probabilities(dataframe=self.gauss_test_results,
                                name='Gaussian Process EMA', exponential=True)
        print('\t|--Plotting XGBoost test results...')
        self.plot_probabilities(dataframe=self.xgboost_test_results,
                                name='XGBoost', exponential=False)
        self.plot_probabilities(dataframe=self.xgboost_test_results,
                                name='XGBoost EMA', exponential=True)
        print('\t|--Plotting Grand Average test results...')
        self.average_model_outputs()
        self.plot_probabilities(dataframe=self.average_model,
                                name='Grand Average', exponential=False)
        self.plot_probabilities(dataframe=self.average_model,
                                name='Grand Average EMA', exponential=True)
        print('\t|--Plotting Weighted Average test results...')
        self.plot_probabilities(dataframe=self.weighted_average_test_results,
                                name='Weighted Average', exponential=False)
        self.plot_probabilities(dataframe=self.weighted_average_test_results,
                                name='Weighted Average EMA', exponential=True)
        
        print('\nPlotted results saved to {}'.format(path.test_results_plots))
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