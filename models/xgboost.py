"""
This module runs an XGBoost model.
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss


class XGBoost:
    """
    Methods and attributes to run an XGBoost model.
    """

    
    def __init__(self):
        """
        depth_range: range of depth values to use during grid-search
        
        child_weight_range: range of child_weight values to use during grid-search
        
        lambda_range: range of lambda values to use during grid-search
        """
        self.cv_indices = []
        self.pred_indices = []
        self.training_y = pd.DataFrame()
        self.testing_y = pd.DataFrame()
        self.full_df = pd.DataFrame()
        self.log_loss_weights = []
        self.feature_names = []
        self.feature_dict = {}
        self.xgboost_optimal_params = {}
        self.xgboost_pred_error = -1
        self.xgboost_predictions = {}
        self.xgboost_cv_predictions = {}
        self.output_name = ''
        self.optimal_depth = -1
        self.optimal_child_weight = -1
        self.optimal_lambda = -1
        self.best_cv_score = 100000
        self.depth_range = [1, 2, 3]
        self.child_weight_range = [2.5, 2, 1.5, 1, 0.5, 0.25, 0.1, 0.05, 0.025,
                                   0.01, 0.005, 0.0025, 0.001]
        self.lambda_range = [0.001, 0.005, 0.010, 0.02, 0.03, 0.4, 0.05,
                            0.06, 0.07, 0.08, 0.09, 0.1, 0.2,
                            0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                            1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5,
                            7, 7.5, 8, 8.5, 9, 9.5, 10]
        self.metadata = {}
        self.importances = []


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


    def cv_depth_weight(self):
        """
        Runs cross-validation by grid-searching through depth and child_weight values.
        """
        from xgboost import XGBClassifier
        
        for depth in self.depth_range:
            for child_weight in self.child_weight_range:
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
                testing_x = self.full_df[self.feature_names].loc[self.cv_indices]
                testing_x_scaled = scaler.transform(testing_x)
                xgboost = XGBClassifier(max_depth=depth,
                                        min_child_weight=child_weight,
                                        gamma=0, learning_rate=0.1,
                                        n_estimators=100, reg_lambda=0.01,
                                        reg_alpha=0, subsample=1,
                                        colsample_bytree=1,
                                        objective='binary:logistic',
                                        booster='gbtree', silent=True,
                                        random_state=123)
                xgboost.fit(X=training_x_scaled, y=self.training_y)
                
                self.testing_y = self.full_df[self.output_name].loc[self.cv_indices]
                self.calculate_log_loss_weights()
                predicted_probs = pd.DataFrame(xgboost.predict_proba(testing_x_scaled))
                all_predicted_probs = all_predicted_probs.append(predicted_probs,
                                                                 ignore_index=True)
                all_testing_y = all_testing_y.append(self.testing_y)
                dates.extend(self.full_df['Dates'].loc[self.cv_indices])
                    
                log_loss_score = log_loss(y_true=all_testing_y,
                                          y_pred=all_predicted_probs,
                                          sample_weight=self.log_loss_weights)
                if log_loss_score < self.best_cv_score:
                    self.best_cv_score = log_loss_score
                    self.optimal_depth = depth
                    self.optimal_child_weight = child_weight
                    self.xgboost_cv_predictions['Dates'] = dates
                    self.xgboost_cv_predictions['True'] = all_testing_y.to_list()
                    self.xgboost_cv_predictions['Predicted'] = all_predicted_probs[1].to_list()


    def cv_lambda(self):
        """
        Runs cross-validation by grid-searching through reg_lambda values.
        """
        from xgboost import XGBClassifier
        
        for reg_lambda in self.lambda_range:
            all_predicted_probs = pd.DataFrame()
            all_testing_y = pd.Series()
            self.log_loss_weights = []
            training_x = self.full_df.loc[: (self.cv_indices[0] - 1),
                                          self.feature_names]
            self.training_y = self.full_df.loc[: (self.cv_indices[0] - 1),
                                               self.output_name]
            scaler = StandardScaler()
            scaler.fit(training_x)
            training_x_scaled = scaler.transform(training_x)
            testing_x = self.full_df[self.feature_names].loc[self.cv_indices]
            testing_x_scaled = scaler.transform(testing_x)
            xgboost = XGBClassifier(max_depth=self.optimal_depth,
                                    min_child_weight=self.optimal_child_weight,
                                    gamma=0, learning_rate=0.1,
                                    n_estimators=100, reg_lambda=reg_lambda,
                                    reg_alpha=0, subsample=1,
                                    colsample_bytree=1,
                                    objective='binary:logistic',
                                    booster='gbtree', silent=True,
                                    random_state=123)
            xgboost.fit(X=training_x_scaled, y=self.training_y)
            feature_importances = pd.DataFrame(xgboost.feature_importances_).T
            feature_importances.rename(columns=self.feature_dict, inplace=True)
            
            self.testing_y = self.full_df[self.output_name].loc[self.cv_indices]
            self.calculate_log_loss_weights()
            predicted_probs = pd.DataFrame(xgboost.predict_proba(testing_x_scaled))
            all_predicted_probs = all_predicted_probs.append(predicted_probs,
                                                             ignore_index=True)
            all_testing_y = all_testing_y.append(self.testing_y)
                
            log_loss_score = log_loss(y_true=all_testing_y,
                                      y_pred=all_predicted_probs,
                                      sample_weight=self.log_loss_weights)
            if log_loss_score <= self.best_cv_score:
                self.best_cv_score = log_loss_score
                self.optimal_lambda = reg_lambda
                self.importances = feature_importances


    def run_xgboost_cv(self):
        """
        Runs the staged cross-validation process for XGBoost.
        """
        self.cv_depth_weight()
        self.cv_lambda()
                    
        self.xgboost_optimal_params['Depth'] = self.optimal_depth
        self.xgboost_optimal_params['Min Child Weight'] = self.optimal_child_weight
        self.xgboost_optimal_params['Lambda'] = self.optimal_lambda
        self.xgboost_optimal_params['Best CV Score'] = self.best_cv_score
        self.metadata['Importances'] = self.importances.to_dict()
        
        
    def run_xgboost_prediction(self):
        """
        Performs prediction on the hold-out sample.
        """
        from xgboost import XGBClassifier
        
        self.optimal_depth = self.xgboost_optimal_params['Depth']
        self.optimal_child_weight = self.xgboost_optimal_params['Min Child Weight']
        self.optimal_lambda = self.xgboost_optimal_params['Lambda']
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
        xgboost = XGBClassifier(max_depth=self.optimal_depth,
                                min_child_weight=self.optimal_child_weight,
                                gamma=0, learning_rate=0.1,
                                n_estimators=100, reg_lambda=self.optimal_lambda,
                                reg_alpha=0, subsample=1,
                                colsample_bytree=1,
                                objective='binary:logistic',
                                booster='gbtree', silent=True,
                                random_state=123)
        xgboost.fit(X=training_x_scaled, y=self.training_y)
        self.importances = pd.DataFrame(xgboost.feature_importances_).T
        self.importances.rename(columns=self.feature_dict, inplace=True)

        testing_x = self.full_df[self.feature_names].loc[self.pred_indices]
        testing_x_scaled = scaler.transform(testing_x)
        self.testing_y = self.full_df[self.output_name].loc[self.pred_indices]
        self.calculate_log_loss_weights()
        predicted_probs = pd.DataFrame(xgboost.predict_proba(testing_x_scaled))
        all_predicted_probs = all_predicted_probs.append(predicted_probs,
                                                         ignore_index=True)
        all_testing_y = all_testing_y.append(self.testing_y)
        dates.extend(self.full_df['Dates'].loc[self.pred_indices])
            
        self.xgboost_pred_error = log_loss(y_true=all_testing_y,
                                           y_pred=all_predicted_probs,
                                           sample_weight=self.log_loss_weights)
        self.xgboost_predictions['Dates'] = dates
        self.xgboost_predictions['True'] = all_testing_y.to_list()
        self.xgboost_predictions['Predicted'] = all_predicted_probs[1].to_list()
        self.metadata['Importances'] = self.importances.to_dict()
        
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
