import numpy as np
import logging
from ..utils import metrics_dict_array


log = logging.getLogger(__name__)


def tune_threshold(y_true, y_scores, metric='f1_score', metric_kwargs=None,
                   always_positive=False, always_negative=False,
                   threshold_minimum=0.0, threshold_maximum=1.0,
                   threshold_stepsize=0.005, bigger_than=True, max_strategy='mean',
                   tuning_fun=None):
    """Find the optimal threshold for a given optimization strategy."""

    # Ensure that both always_positive and always_negative are not True simultaneously
    assert not (always_positive and always_negative), "always_positive and always_negative cannot both be true"

    # Handle special cases
    if always_positive:
        return {'threshold': -0.001, 'index': 0, 'performance': 0}, None, None
    elif always_negative:
        return {'threshold': 1.001, 'index': -1, 'performance': 0}, None, None

    if metric_kwargs is None:
        metric_kwargs = {}

    # Generate an array of thresholds
    thresholds = np.arange(threshold_minimum, threshold_maximum + threshold_stepsize, threshold_stepsize)

    # Vectorized threshold comparison
    if bigger_than:
        binary_predictions = (y_scores[:, np.newaxis] > thresholds).astype(int)
    else:
        binary_predictions = (y_scores[:, np.newaxis] <= thresholds).astype(int)

    # If a custom tuning function is provided, use it
    if tuning_fun is not None:
        tuning_function = tuning_fun
    else:
        tuning_function = metrics_dict_array[metric]
    
    # Compute performance for each threshold
    performances = np.array([tuning_function(y_true, binary_predictions[:, i], **metric_kwargs) for i in range(binary_predictions.shape[1])])

    if (performances.size == 0) or (np.isnan(performances).all()):
        return {'threshold': 1.001, 'index': -1, 'performance': 0}, None, None
    
    # Find all indices where the performance is maximized
    max_value = np.nanmax(performances)
    max_indexes = np.where(performances == max_value)[0]  # Indices of all max values

    # Define how to handle multiple max-values based on the input `max_strategy`
    if max_strategy == 'min':
        max_index = max_indexes.min()
    elif max_strategy == 'max':
        max_index = max_indexes.max()
    elif max_strategy == 'mean':
        max_index = int(max_indexes.mean())  # Mean index, convert to integer
    else:
        raise ValueError("Invalid value for max_strategy. Choose 'min', 'max', or 'mean'.")
    
    best_threshold = thresholds[max_index]
    best_performance = performances[max_index]
    
    return {
        'threshold': best_threshold,
        'index': max_index,
        'performance': best_performance
    }, thresholds, performances
