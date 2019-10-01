"""
This module runs a deployment version of an SVM model.
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss


class SupportVectorMachine:
    """
    Methods and attributes to run an Elastic Net model.
    """

    
    def __init__(self):
        """
        C_range: range of of C values to use during grid-search
        
        gamma_range: range of gamma values to use during grid-search
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
        self.svm_optimal_params = {}
        self.svm_pred_error = -1
        self.svm_predictions = {}
        self.svm_cv_predictions = {}
        self.output_name = ''
        self.optimal_C = -1
        self.optimal_gamma = -1
        self.best_cv_score = 100000
        self.C_range = [0.0001, 0.00025, 0.0005, 0.00075, 0.001, 0.0025, 0.005,
                        0.0075, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75,
                        1.0, 2.5, 5.0, 7.5, 10.0]
        self.gamma_range = []
        self.metadata = {}
        self.support_vector_count_as_percent = -1


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


    def run_svm_cv(self):
        """
        Runs cross-validation by grid-searching through C and gamma values.
        """
        from sklearn.svm import SVC
        
        default_gamma = 1 / len(self.feature_names)
        self.gamma_range = [multiplier * default_gamma
                            for multiplier in [0.25, 0.50, 0.75, 1.0, 1.25,
                                               1.50, 1.75, 2.00]]
        for C in self.C_range:
            for gamma in self.gamma_range:
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
                    svm = SVC(C=C, kernel='rbf', gamma=gamma, probability=True,
                              tol=1e-3, random_state=123,
                              class_weight='balanced')
                    svm.fit(X=training_x_scaled, y=self.training_y)
                    svm_count = len(svm.support_) / len(training_x_scaled)
                    testing_x = self.full_df[self.feature_names].loc[self.cv_indices]
                    testing_x_scaled = scaler.transform(testing_x)
                    self.testing_y = self.full_df[self.output_name].loc[self.cv_indices]
                    self.calculate_log_loss_weights()
                    
                    predicted_probs = pd.DataFrame(svm.predict_proba(X=testing_x_scaled))
                    all_predicted_probs = all_predicted_probs.append(predicted_probs,
                                                                     ignore_index=True)
                    all_testing_y = all_testing_y.append(self.testing_y)
                    dates.extend(self.full_df['Dates'].loc[self.cv_indices])
                        
                log_loss_score = log_loss(y_true=all_testing_y,
                                          y_pred=all_predicted_probs,
                                          sample_weight=self.log_loss_weights)
                if log_loss_score < self.best_cv_score:
                    self.best_cv_score = log_loss_score
                    self.optimal_C = C
                    self.optimal_gamma = gamma
                    self.support_vector_count_as_percent = round(svm_count, 3)
                    self.svm_cv_predictions['Dates'] = dates
                    self.svm_cv_predictions['True'] = all_testing_y.to_list()
                    self.svm_cv_predictions['Predicted'] = all_predicted_probs[1].to_list()
            
        self.svm_optimal_params['C'] = self.optimal_C
        self.svm_optimal_params['Gamma'] = self.optimal_gamma
        self.svm_optimal_params['Best CV Score'] = self.best_cv_score
        self.metadata['SV Count %'] = self.support_vector_count_as_percent
        
        
    def run_svm_prediction(self):
        """
        Performs prediction on the hold-out sample.
        """
        from sklearn.svm import SVC
        
        self.optimal_C = self.svm_optimal_params['C']
        self.optimal_gamma = self.svm_optimal_params['Gamma']
        all_predicted_probs = pd.DataFrame()
        all_testing_y = pd.Series()
        dates = []
        training_x = self.full_df.loc[: (self.pred_indices[0] - 1),
                                      self.feature_names]
        self.training_y = self.full_df.loc[: (self.pred_indices[0] - 1),
                                           self.output_name]
        scaler = StandardScaler()
        scaler.fit(training_x)
        training_x_scaled = scaler.transform(training_x)
        svm = SVC(C=self.optimal_C, kernel='rbf', gamma=self.optimal_gamma,
                  probability=True, tol=1e-3, random_state=123,
                  class_weight='balanced')
        svm.fit(X=training_x_scaled, y=self.training_y)
        self.support_vector_count_as_percent = len(svm.support_) / len(training_x_scaled)

        testing_x = self.full_df[self.feature_names].loc[self.pred_indices]
        testing_x_scaled = scaler.transform(testing_x)
        self.testing_y = self.full_df[self.output_name].loc[self.pred_indices]
        predicted_probs = pd.DataFrame(svm.predict_proba(X=testing_x_scaled))
        all_predicted_probs = all_predicted_probs.append(predicted_probs,
                                                         ignore_index=True)
        all_testing_y = all_testing_y.append(self.testing_y)
        dates.extend(self.full_df['Dates'].loc[self.pred_indices])
            
        self.svm_predictions['Dates'] = dates
        self.svm_predictions['True'] = all_testing_y.to_list()
        self.svm_predictions['Predicted'] = all_predicted_probs[1].to_list()
        self.metadata['SV Count %'] = round(self.support_vector_count_as_percent, 3)


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