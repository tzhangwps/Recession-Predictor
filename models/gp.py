"""
This module runs a Gaussian Process model.
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss


class GaussianProcess:
    """
    Methods and attributes to run a Gaussian Process model.
    """

    
    def __init__(self):
        self.cv_params = {}
        self.cv_start = ''
        self.cv_end = ''
        self.test_name = ''
        self.cv_indices = []
        self.pred_indices = []
        self.training_y = pd.DataFrame()
        self.testing_y = pd.DataFrame()
        self.full_df = pd.DataFrame()
        self.log_loss_weights = []
        self.feature_names = []
        self.gauss_optimal_params = {}
        self.gauss_pred_error = -1
        self.gauss_predictions = {}
        self.gauss_cv_predictions = {}
        self.output_name = ''
        self.length_scale_range = (1e-05, 100000.0)
        self.alpha_range = (1e-05, 100000.0)
        self.metadata = {}
        self.length_scale = -1
        self.alpha = -1


    def calculate_log_loss_weights(self):
        """
        Calculates weight adjustments for class outputs, such that each class
        receives the same weight in log loss calculations.
        """
        true_output_labels = self.training_y.unique()
        desired_weight = 1 / len(true_output_labels)
        
        class_weights = {}
        for label in true_output_labels:
            training_frequency = (len(self.training_y[self.training_y == label])
                / len(self.training_y))
            multiplier = desired_weight / training_frequency
            class_weights[str(label)] = multiplier
        
        for sample in self.testing_y:
            self.log_loss_weights.append(class_weights[str(sample)])
        
    
    def get_cv_indices(self):
        """
        Gets indices for rows to be used during cross-validation.
        """
        if self.full_df['Dates'][0] > self.full_df['Dates'][len(self.full_df) - 1]:
            self.full_df = self.full_df[::-1]
        self.full_df.reset_index(inplace=True)
        self.full_df.drop('index', axis=1, inplace=True)
        date_condition = ((self.full_df['Dates'] <= self.cv_end) &
                          (self.full_df['Dates'] >= self.cv_start))
        self.cv_indices = list(self.full_df[date_condition].index)
    
    
    def run_gauss_cv(self):
        """
        Runs cross-validation to generate cross-validation errors.
        Hyperparameters are automatically tuned.
        """
        from sklearn.gaussian_process import GaussianProcessClassifier
        from sklearn.gaussian_process.kernels import RationalQuadratic
        
        all_predicted_probs = pd.DataFrame()
        all_testing_y = pd.Series()
        dates = []
        self.log_loss_weights = []
        for test_name in range(1, self.test_name + 1):
            self.cv_start = self.cv_params[test_name]['cv_start']
            self.cv_end = self.cv_params[test_name]['cv_end']
            self.get_cv_indices()
            training_x = self.full_df.loc[: (self.cv_indices[0] - 1),
                                          self.feature_names]
            self.training_y = self.full_df.loc[: (self.cv_indices[0] - 1),
                                               self.output_name]
            scaler = StandardScaler()
            scaler.fit(training_x)
            training_x_scaled = scaler.transform(training_x)
            rational_quadratic = RationalQuadratic(length_scale_bounds=self.length_scale_range,
                                                   alpha_bounds=self.alpha_range)
            gauss = GaussianProcessClassifier(kernel=rational_quadratic,
                                              max_iter_predict=100,
                                              random_state=123)
            gauss.fit(X=training_x_scaled, y=self.training_y)
            self.length_scale = gauss.kernel_.length_scale
            self.alpha = gauss.kernel_.alpha
    
            testing_x = self.full_df[self.feature_names].loc[self.cv_indices]
            testing_x_scaled = scaler.transform(testing_x)
            self.testing_y = self.full_df[self.output_name].loc[self.cv_indices]
            self.calculate_log_loss_weights()
            predicted_probs = pd.DataFrame(gauss.predict_proba(X=testing_x_scaled))
            all_predicted_probs = all_predicted_probs.append(predicted_probs,
                                                             ignore_index=True)
            all_testing_y = all_testing_y.append(self.testing_y)
            dates.extend(self.full_df['Dates'].loc[self.cv_indices])
                
        self.gauss_cv_error = log_loss(y_true=all_testing_y,
                                         y_pred=all_predicted_probs,
                                         sample_weight=self.log_loss_weights)
        self.gauss_cv_predictions['Dates'] = dates
        self.gauss_cv_predictions['True'] = all_testing_y.to_list()
        self.gauss_cv_predictions['Predicted'] = all_predicted_probs[1].to_list()
        self.gauss_optimal_params['Best CV Score'] = self.gauss_cv_error
        self.metadata['Length Scale'] = self.length_scale
        self.metadata['Alpha'] = self.alpha        
    
    
    def run_gauss_prediction(self):
        """
        Performs prediction on the hold-out sample.
        """
        from sklearn.gaussian_process import GaussianProcessClassifier
        from sklearn.gaussian_process.kernels import RationalQuadratic
        
        all_predicted_probs = pd.DataFrame()
        all_testing_y = pd.Series()
        dates = []
        self.log_loss_weights = []
        training_x = self.full_df.loc[: (self.pred_indices[0] - 1),
                                      self.feature_names]
        self.training_y = self.full_df.loc[: (self.pred_indices[0] - 1),
                                           self.output_name]
        scaler = StandardScaler()
        scaler.fit(training_x)
        training_x_scaled = scaler.transform(training_x)
        rational_quadratic = RationalQuadratic(length_scale_bounds=self.length_scale_range,
                                               alpha_bounds=self.alpha_range)
        gauss = GaussianProcessClassifier(kernel=rational_quadratic,
                                          max_iter_predict=100,
                                          random_state=123)
        gauss.fit(X=training_x_scaled, y=self.training_y)
        self.length_scale = gauss.kernel_.length_scale
        self.alpha = gauss.kernel_.alpha

        testing_x = self.full_df[self.feature_names].loc[self.pred_indices]
        testing_x_scaled = scaler.transform(testing_x)
        self.testing_y = self.full_df[self.output_name].loc[self.pred_indices]
        self.calculate_log_loss_weights()
        predicted_probs = pd.DataFrame(gauss.predict_proba(X=testing_x_scaled))
        all_predicted_probs = all_predicted_probs.append(predicted_probs,
                                                         ignore_index=True)
        all_testing_y = all_testing_y.append(self.testing_y)
        dates.extend(self.full_df['Dates'].loc[self.pred_indices])
            
        self.gauss_pred_error = log_loss(y_true=all_testing_y,
                                         y_pred=all_predicted_probs,
                                         sample_weight=self.log_loss_weights)
        self.gauss_predictions['Dates'] = dates
        self.gauss_predictions['True'] = all_testing_y.to_list()
        self.gauss_predictions['Predicted'] = all_predicted_probs[1].to_list()
        self.metadata['Length Scale'] = self.length_scale
        self.metadata['Alpha'] = self.alpha
        
        
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