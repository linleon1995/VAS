from typing import Dict

import mlflow


class MlflowTracker:
    def __init__(self):
        pass

    def run(self, step, train_loss, valid_loss, valid_metrics: Dict = None):
        mlflow.log_metric('train_loss', train_loss, step=step)
        mlflow.log_metric('valid_loss', valid_loss, step=step)
        if valid_metrics is not None:
            for metrics, value in valid_metrics.items():
                mlflow.log_metric(metrics, value, step=step)
