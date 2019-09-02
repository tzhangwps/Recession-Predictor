"""
This module deploys the chosen model.
"""
import json
import pandas as pd
from datetime import datetime

import RecessionPredictor_paths as path
from models.deployment_svm import SupportVectorMachine


class CrossValidate:
    """
    Methods and attributes for cross-validation.
    """
    
    
    def __init__(self):
        self.cv_params = {}
        self.test_name = ''
        self.full_df = pd.DataFrame()
        self.cv_indices = []
        self.feature_names = []
        self.feature_dict = {}
        self.output_names = []
        self.optimal_params_by_output = {}
        self.cv_metadata_by_output = {}
        self.cv_predictions_by_output = {}

                
    def walk_forward_cv(self):
        """
        Runs walk-forward cross-validation, and saves cross-validation
        metrics.
        """
        for output_name in self.output_names:
            print('\t\t\t|--Prediction type: {}'.format(output_name))
            optimal_params_by_model = {}
            cv_metadata_by_model = {}
            cv_predictions_by_model = {}
            
            print('\t\t\t\t|--SVM Model')
            svm = SupportVectorMachine()
            svm.cv_params = self.cv_params
            svm.test_name = self.test_name
            svm.full_df = self.full_df
            svm.feature_names = self.feature_names
            svm.output_name = output_name
            svm.run_svm_cv()
            optimal_params_by_model['SVM'] = svm.svm_optimal_params
            cv_metadata_by_model['SVM'] = svm.metadata
            cv_predictions_by_model['SVM'] = svm.svm_cv_predictions
            
            self.optimal_params_by_output[output_name] = optimal_params_by_model
            self.cv_metadata_by_output[output_name] = cv_metadata_by_model
            self.cv_predictions_by_output[output_name] = cv_predictions_by_model


class Predict:
    """
    Methods and attributes for prediction.
    """
    
    
    def __init__(self):
        self.pred_start = ''
        self.pred_end = ''
        self.full_df = pd.DataFrame()
        self.pred_indices = []
        self.feature_names = []
        self.feature_dict = {}
        self.output_names = []
        self.predictions_by_output = {}
        self.cv_predictions_by_output = {}
        self.optimal_params_by_output = {}
        self.pred_metadata_by_output = {}
        
        
    def get_prediction_indices(self):
        """
        Gets indices for rows to be used during prediction.
        """
        if self.full_df['Dates'][0] > self.full_df['Dates'][len(self.full_df) - 1]:
            self.full_df = self.full_df[::-1]
        self.full_df.reset_index(inplace=True)
        self.full_df.drop('index', axis=1, inplace=True)
        date_condition = ((self.full_df['Dates'] <= self.pred_end) &
                          (self.full_df['Dates'] >= self.pred_start))
        self.pred_indices = list(self.full_df[date_condition].index)
        
        
    def walk_forward_prediction(self):
        """
        Runs walk-forward prediction, and saves prediction metrics.
        """
        for output_name in self.output_names:
            print('\t\t\t|--Prediction type: {}'.format(output_name))
            predictions_by_model = {}
            pred_metadata_by_model = {}
            
            print('\t\t\t\t|--SVM Model')
            svm = SupportVectorMachine()
            svm.pred_indices = self.pred_indices
            svm.full_df = self.full_df
            svm.feature_names = self.feature_names
            svm.output_name = output_name
            svm.svm_optimal_params = self.optimal_params_by_output[output_name]['SVM']
            svm.run_svm_prediction()
            predictions_by_model['SVM'] = svm.svm_predictions
            pred_metadata_by_model['SVM'] = svm.metadata
            
            self.predictions_by_output[output_name] = predictions_by_model
            self.pred_metadata_by_output[output_name] = pred_metadata_by_model
        
        
    def run_prediction(self):
        """
        Gets indices for rows to be used during prediction, and performs
        walk-forward prediction.
        """
        self.get_prediction_indices()
        self.walk_forward_prediction()



class Deployer:
    """
    The manager class for this module.
    """
    
    
    def __init__(self):
        self.final_df_output = pd.DataFrame()
        self.testing_dates = {}
        self.optimal_params = {}
        self.cv_model_metadata = {}
        self.pred_model_metadata = {}
        self.full_predictions = {}
        self.feature_names = ['Payrolls_3mo_vs_12mo',
                              'Real_Fed_Funds_Rate_12mo_chg',
                              'CPI_3mo_pct_chg_annualized',
                              '10Y_Treasury_Rate_12mo_chg',
                              '3M_10Y_Treasury_Spread',
                              'S&P_500_12mo_chg']
        self.feature_dict = {0: 'Payrolls_3mo_vs_12mo',
                             1: 'Real_Fed_Funds_Rate_12mo_chg',
                             2: 'CPI_3mo_pct_chg_annualized',
                             3: '10Y_Treasury_Rate_12mo_chg',
                             4: '3M_10Y_Treasury_Spread',
                             5: 'S&P_500_12mo_chg'}
        self.output_names = ['Recession',
                             'Recession_within_6mo',
                             'Recession_within_12mo',
                             'Recession_within_24mo']

    
    def fill_testing_dates(self):
        """
        Stores testing dates, for each test number.
        """
        
        now = datetime.now()
        month = now.strftime('%m')
        year = now.year        
        most_recent_date = '{}-{}-01'.format(year, month)
        self.testing_dates[1] = {'cv_start': '1972-01-01', 
                                 'cv_end': '1975-12-01', 
                                 'pred_start': '1976-01-01',
                                 'pred_end': '1981-07-01'}
        self.testing_dates[2] = {'cv_start': '1976-01-01', 
                                 'cv_end': '1981-07-01', 
                                 'pred_start': '1981-08-01',
                                 'pred_end': '1983-07-01'}
        self.testing_dates[3] = {'cv_start': '1976-01-01', 
                                 'cv_end': '1983-07-01', 
                                 'pred_start': '1983-08-01',
                                 'pred_end': '1992-12-01'}
        self.testing_dates[4] = {'cv_start': '1983-08-01', 
                                 'cv_end': '1992-12-01', 
                                 'pred_start': '1993-01-01',
                                 'pred_end': '2003-07-01'}
        self.testing_dates[5] = {'cv_start': '1993-01-01', 
                                 'cv_end': '2003-07-01', 
                                 'pred_start': '2003-08-01',
                                 'pred_end': '2010-09-01'}
        self.testing_dates[6] = {'cv_start': '2003-08-01', 
                                 'cv_end': '2010-09-01', 
                                 'pred_start': '2010-10-01',
                                 'pred_end': most_recent_date}
    
    
    def perform_backtests(self):
        """
        Performs cross-validation and prediction.
        """
        
        for test_name in self.testing_dates:
            print('\t|--Test #{}'.format(test_name))
            test_dates = self.testing_dates[test_name]
            print('\t\t|--Performing Nested Cross-Validation')
            cross_validation = CrossValidate()
            cross_validation.output_names = self.output_names
            cross_validation.feature_names = self.feature_names
            cross_validation.feature_dict = self.feature_dict
            cross_validation.full_df = self.final_df_output
            cross_validation.cv_params = self.testing_dates
            cross_validation.test_name = test_name
            cross_validation.walk_forward_cv()
            self.optimal_params['Test #{}'.format(test_name)] = cross_validation.optimal_params_by_output
            self.cv_model_metadata['Test #{}'.format(test_name)] = cross_validation.cv_metadata_by_output
            
            print('\t\t|--Performing Out-Of-Sample Testing')
            prediction = Predict()
            prediction.output_names = self.output_names
            prediction.feature_names = self.feature_names
            prediction.feature_dict = self.feature_dict
            prediction.optimal_params_by_output = cross_validation.optimal_params_by_output
            prediction.cv_predictions_by_output = cross_validation.cv_predictions_by_output
            prediction.full_df = self.final_df_output
            prediction.pred_start = test_dates['pred_start']
            prediction.pred_end = test_dates['pred_end']
            prediction.run_prediction()
            self.full_predictions['Test #{}'.format(test_name)] = prediction.predictions_by_output
            self.pred_model_metadata['Test #{}'.format(test_name)] = prediction.pred_metadata_by_output
        
        print('\nSaving model metadata...')
        with open(path.deployment_cv_results, 'w') as file:
            json.dump(self.optimal_params, file)
        with open(path.deployment_cv_metadata, 'w') as file:
            json.dump(self.cv_model_metadata, file)
        with open(path.deployment_pred_model_metadata, 'w') as file:
            json.dump(self.pred_model_metadata, file)
        with open(path.deployment_full_predictions, 'w') as file:
            json.dump(self.full_predictions, file)
    

    def read_full_predictions(self, model_name):
        """
        Given a specific model, loops through each Test to save model predictions
        into a single dataframe.
        
        model_name: name of the model.
        """
        dates = []
        true_0mo = []
        true_6mo = []
        pred_6mo = []
        true_12mo = []
        pred_12mo = []
        true_24mo = []
        pred_24mo = []
        for test in self.full_predictions:
            test_data = self.full_predictions[test]
            dates.extend(test_data[self.output_names[0]][model_name]['Dates'])
            true_0mo.extend(test_data[self.output_names[0]][model_name]['True'])
            true_6mo.extend(test_data[self.output_names[1]][model_name]['True'])
            pred_6mo.extend(test_data[self.output_names[1]][model_name]['Predicted'])
            true_12mo.extend(test_data[self.output_names[2]][model_name]['True'])
            pred_12mo.extend(test_data[self.output_names[2]][model_name]['Predicted'])
            true_24mo.extend(test_data[self.output_names[3]][model_name]['True'])
            pred_24mo.extend(test_data[self.output_names[3]][model_name]['Predicted'])
                
        results = pd.DataFrame()   
        results['Dates'] = dates
        results['True_{}'.format(self.output_names[0])] = true_0mo
        results['True_{}'.format(self.output_names[1])] = true_6mo
        results['Pred_{}'.format(self.output_names[1])] = pred_6mo
        results['True_{}'.format(self.output_names[2])] = true_12mo
        results['Pred_{}'.format(self.output_names[2])] = pred_12mo
        results['True_{}'.format(self.output_names[3])] = true_24mo
        results['Pred_{}'.format(self.output_names[3])] = pred_24mo
        
        return(results)
            
    
    def create_full_predictions_dataframe(self):
        """
        Organizes predictions for the chosen model into a single dataframe.
        """
        print('\nSaving Full Predictions as dataframes...')
        with open(path.deployment_full_predictions, 'r') as file:
            self.full_predictions = json.load(file)
        self.read_full_predictions('SVM').to_json(path.deployment_svm_test_results)
        print('\t|--SVM results saved to {}'.format(path.deployment_svm_test_results))
        
    
    def run_test_procedures(self):
        """
        Runs test procedures on final dataset.
        """
        print('\nDeploying prediction model...\n')
        self.final_df_output = pd.read_json(path.data_final)
        self.final_df_output.sort_index(inplace=True)
        self.fill_testing_dates()
        self.perform_backtests()
        self.create_full_predictions_dataframe()
        print('\nDeployment complete!')
        
        
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