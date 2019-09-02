"""
This module runs an Elastic Net model.
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss


class ElasticNet:
    """
    Methods and attributes to run an Elastic Net model.
    """

    
    def __init__(self):
        """
        alpha_range: range of alpha values to use during grid-search
        
        l1_ratio_range: range of l1_ratio values to use during grid-search
        """
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
        self.feature_dict = {}
        self.elastic_net_optimal_params = {}
        self.elastic_net_pred_error = -1
        self.elastic_net_predictions = {}
        self.elastic_net_cv_predictions = {}
        self.output_name = ''
        self.optimal_alpha = -1
        self.optimal_l1_ratio = -1
        self.best_cv_score = 100000
        self.alpha_range = [0.001, 0.005, 0.010, 0.020, 0.030, 0.040, 0.050,
                            0.060, 0.070, 0.080, 0.090, 0.095, 0.100, 0.110,
                            0.120, 0.130, 0.140, 0.150, 0.160, 0.170, 0.180,
                            0.190, 0.200, 0.210, 0.220, 0.230, 0.240, 0.250,
                            0.260, 0.270, 0.280, 0.290, 0.300, 0.310, 0.320,
                            0.330, 0.340, 0.350, 0.360, 0.370, 0.380, 0.390,
                            0.400, 0.410, 0.420, 0.430, 0.440, 0.450, 0.460,
                            0.470, 0.480, 0.490, 0.500, 0.600, 0.700, 0.800,
                            0.900, 1.000]
        self.l1_ratio_range = [0]
        self.metadata = {}
        self.coefficients = []


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


    def run_elastic_net_cv(self):
        """
        Runs cross-validation by grid-searching through alpha and l1_ratio values.
        """
        from sklearn.linear_model import SGDClassifier
        
        for alpha in self.alpha_range:
            for l1_ratio in self.l1_ratio_range:
                all_predicted_probs = pd.DataFrame()
                all_testing_y = pd.Series()
                dates = []
                self.log_loss_weights = []
                for test_name in range(1, max(self.test_name + 1, 2)):
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
                    testing_x = self.full_df[self.feature_names].loc[self.cv_indices]
                    testing_x_scaled = scaler.transform(testing_x)
                    elastic_net = SGDClassifier(loss='log', penalty='elasticnet',
                                                alpha=alpha, l1_ratio=l1_ratio,
                                                max_iter=1000, tol=1e-3,
                                                random_state=123)
                    elastic_net.fit(X=training_x_scaled, y=self.training_y)
                    
                    self.testing_y = self.full_df[self.output_name].loc[self.cv_indices]
                    self.calculate_log_loss_weights()
    
                    coefficients = pd.DataFrame(elastic_net.coef_)
                    coefficients.rename(columns=self.feature_dict, inplace=True)
                    predicted_probs = pd.DataFrame(elastic_net.predict_proba(X=testing_x_scaled))
                    all_predicted_probs = all_predicted_probs.append(predicted_probs,
                                                                     ignore_index=True)
                    all_testing_y = all_testing_y.append(self.testing_y)   
                    dates.extend(self.full_df['Dates'].loc[self.cv_indices])
                    
                log_loss_score = log_loss(y_true=all_testing_y,
                                          y_pred=all_predicted_probs,
                                          sample_weight=self.log_loss_weights)
                if log_loss_score < self.best_cv_score:
                    self.best_cv_score = log_loss_score
                    self.optimal_alpha = alpha
                    self.optimal_l1_ratio = l1_ratio
                    self.coefficients = coefficients.to_dict()
                    self.elastic_net_cv_predictions['Dates'] = dates
                    self.elastic_net_cv_predictions['True'] = all_testing_y.to_list()
                    self.elastic_net_cv_predictions['Predicted'] = all_predicted_probs[1].to_list()
            
        self.elastic_net_optimal_params['Alpha'] = self.optimal_alpha
        self.elastic_net_optimal_params['L1_Ratio'] = self.optimal_l1_ratio
        self.elastic_net_optimal_params['Best CV Score'] = self.best_cv_score
        self.metadata['Coefficients'] = self.coefficients
        
        
    def run_elastic_net_prediction(self):
        """
        Performs prediction on the hold-out sample.
        """
        from sklearn.linear_model import SGDClassifier
        
        self.optimal_alpha = self.elastic_net_optimal_params['Alpha']
        self.optimal_l1_ratio = self.elastic_net_optimal_params['L1_Ratio']
        all_predicted_probs = pd.DataFrame()
        all_testing_y = pd.Series()
        dates = []
        self.log_loss_weights = []
        training_x = self.full_df.loc[: (self.pred_indices[0] - 1),
                                      self.feature_names]
        scaler = StandardScaler()
        scaler.fit(training_x)
        training_x_scaled = scaler.transform(training_x)
        self.training_y = self.full_df.loc[: (self.pred_indices[0] - 1),
                                           self.output_name]
        elastic_net = SGDClassifier(loss='log', penalty='elasticnet',
                                    alpha=self.optimal_alpha,
                                    l1_ratio=self.optimal_l1_ratio,
                                    max_iter=1000, tol=1e-3,
                                    random_state=123)
        elastic_net.fit(X=training_x_scaled, y=self.training_y)
        self.coefficients = pd.DataFrame(elastic_net.coef_)
        self.coefficients.rename(columns=self.feature_dict, inplace=True)

        testing_x = self.full_df[self.feature_names].loc[self.pred_indices]
        testing_x_scaled = scaler.transform(testing_x)
        self.testing_y = self.full_df[self.output_name].loc[self.pred_indices]
        self.calculate_log_loss_weights()
        predicted_probs = pd.DataFrame(elastic_net.predict_proba(X=testing_x_scaled))
        all_predicted_probs = all_predicted_probs.append(predicted_probs,
                                                         ignore_index=True)
        all_testing_y = all_testing_y.append(self.testing_y)
        dates.extend(self.full_df['Dates'].loc[self.pred_indices])
            
        self.elastic_net_pred_error = log_loss(y_true=all_testing_y,
                                               y_pred=all_predicted_probs,
                                               sample_weight=self.log_loss_weights)
        self.elastic_net_predictions['Dates'] = dates
        self.elastic_net_predictions['True'] = all_testing_y.to_list()
        self.elastic_net_predictions['Predicted'] = all_predicted_probs[1].to_list()
        self.metadata['Coefficients'] = self.coefficients.to_dict()
        
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