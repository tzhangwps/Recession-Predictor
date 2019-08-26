"""
This module runs a Naive Bayes model.
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss


class NaiveBayes:
    """
    Methods and attributes to run a Naive Bayes model.
    """

    
    def __init__(self):
        self.cv_indices = []
        self.pred_indices = []
        self.training_y = pd.DataFrame()
        self.testing_y = pd.DataFrame()
        self.full_df = pd.DataFrame()
        self.log_loss_weights = []
        self.feature_names = []
        self.bayes_optimal_params = {}
        self.bayes_pred_error = -1
        self.bayes_predictions = {}
        self.bayes_cv_predictions = {}
        self.output_name = ''


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
    
    
    def run_bayes_cv(self):
        """
        Runs cross-validation to generate cross-validation errors.
        """
        from sklearn.naive_bayes import GaussianNB
        
        all_predicted_probs = pd.DataFrame()
        all_testing_y = pd.Series()
        dates = []
        self.log_loss_weights = []
        training_x = self.full_df.loc[: (self.cv_indices[0] - 1),
                                      self.feature_names]
        self.training_y = self.full_df.loc[: (self.cv_indices[0] - 1),
                                           self.output_name]
        scaler = StandardScaler()
        scaler.fit(training_x)
        training_x_scaled = scaler.transform(training_x)
        naive_bayes = GaussianNB()
        naive_bayes.fit(X=training_x_scaled, y=self.training_y)

        testing_x = self.full_df[self.feature_names].loc[self.cv_indices]
        testing_x_scaled = scaler.transform(testing_x)
        self.testing_y = self.full_df[self.output_name].loc[self.cv_indices]
        self.calculate_log_loss_weights()
        predicted_probs = pd.DataFrame(naive_bayes.predict_proba(X=testing_x_scaled))
        all_predicted_probs = all_predicted_probs.append(predicted_probs,
                                                         ignore_index=True)
        all_testing_y = all_testing_y.append(self.testing_y)
        dates.extend(self.full_df['Dates'].loc[self.cv_indices])
            
        self.bayes_cv_error = log_loss(y_true=all_testing_y,
                                       y_pred=all_predicted_probs,
                                       sample_weight=self.log_loss_weights)
        self.bayes_cv_predictions['Dates'] = dates
        self.bayes_cv_predictions['True'] = all_testing_y.to_list()
        self.bayes_cv_predictions['Predicted'] = all_predicted_probs[1].to_list()
        self.bayes_optimal_params['Best CV Score'] = self.bayes_cv_error
    
        
    def run_bayes_prediction(self):
        """
        Performs prediction on the hold-out sample.
        """
        from sklearn.naive_bayes import GaussianNB
        
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
        naive_bayes = GaussianNB()
        naive_bayes.fit(X=training_x_scaled, y=self.training_y)

        testing_x = self.full_df[self.feature_names].loc[self.pred_indices]
        testing_x_scaled = scaler.transform(testing_x)
        self.testing_y = self.full_df[self.output_name].loc[self.pred_indices]
        self.calculate_log_loss_weights()
        predicted_probs = pd.DataFrame(naive_bayes.predict_proba(X=testing_x_scaled))
        all_predicted_probs = all_predicted_probs.append(predicted_probs,
                                                         ignore_index=True)
        all_testing_y = all_testing_y.append(self.testing_y)
        dates.extend(self.full_df['Dates'].loc[self.pred_indices])
            
        self.bayes_pred_error = log_loss(y_true=all_testing_y,
                                         y_pred=all_predicted_probs,
                                         sample_weight=self.log_loss_weights)
        self.bayes_predictions['Dates'] = dates
        self.bayes_predictions['True'] = all_testing_y.to_list()
        self.bayes_predictions['Predicted'] = all_predicted_probs[1].to_list()
        

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