"""
This module runs a weighted average model.
"""
from scipy.stats import rankdata

class WeightedAverage:
    """
    Methods and attributes to average the predictions of a group of models.
    """
    
    
    def __init__(self):
        """
        rank_scheme: weights applied to each model rank, where higher ranks
        correspond to models with better cross-validation performance.
        """
        self.model_names = []
        self.model_weights = {}
        self.cv_results = {}
        self.predictions_by_model = {}
        self.weighted_average_predictions = {}
        self.metadata={}
        self.rank_scheme = {1: 0.25,
                            2: 0.25,
                            3: 0.25,
                            4: 0.25,
                            5: 0.00,
                            6: 0.00}
            

    def calculate_model_weights(self):
        """
        Assign weights to each model's prediction.
        """
        cv_scores = []
        for model_name in self.cv_results:
            cv_scores.append(self.cv_results[model_name]['Best CV Score'])
        
        score_ranks = rankdata(cv_scores)
        iteration = 0
        for model_name in self.cv_results:
            self.model_weights[model_name] = self.rank_scheme[int(score_ranks[iteration])]
            iteration += 1
        self.metadata['Weights'] = self.model_weights

    
    def reshape_predictions_by_model(self):
        """
        Creates a dataset that consolidates each model's predictions.
        """
        self.predictions_reshaped = {}
        self.predictions_reshaped['Dates'] = self.predictions_by_model[self.model_names[0]]['Dates']
        self.predictions_reshaped['True'] = self.predictions_by_model[self.model_names[0]]['True']
        for model_name in self.predictions_by_model:
            self.predictions_reshaped[model_name] = self.predictions_by_model[model_name]['Predicted']
        
    
    def weighted_model_predictions(self):
        """
        Creates predictions for the weighted model.
        """
        predicted = []
        for index in range(0, len(self.predictions_reshaped['Dates'])):
            weighted_sum = 0    
            for model_name in self.model_names:
                model_weight = self.model_weights[model_name]
                prediction = self.predictions_reshaped[model_name][index]
                weighted_sum += model_weight * prediction
            predicted.append(weighted_sum)
        self.weighted_average_predictions['Predicted'] = predicted
        
        
    def run_weighted_average_prediction(self):
        """
        Creates a weighted average model by weighting the predictions of
        several models.
        """
        self.log_loss_weights = []
        self.calculate_model_weights()
        self.reshape_predictions_by_model()
        self.weighted_model_predictions()
        self.weighted_average_predictions['Dates'] = self.predictions_reshaped['Dates']
        self.weighted_average_predictions['True'] = self.predictions_reshaped['True']