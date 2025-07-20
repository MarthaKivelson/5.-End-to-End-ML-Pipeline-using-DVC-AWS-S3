import os 
import numpy as np
import pandas as pd
import pickle 
import json 
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import logging
import yaml 
from dvclive import Live


log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('model_eval')
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

log_file_path = os.path.join(log_dir, "model_eval.log")
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel("DEBUG")

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_yaml(params_path: str) -> dict:
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug("Successfully loaded params file")
        return params
    except FileNotFoundError:
        logger.error("Params file not found")
        raise
    except yaml.YAMLError as e:
        logger.error("Yaml error", e)
        raise
    except Exception as e:
        logger.error("Unexpected error while loading yaml file")
        raise


def load_model(file_path: str) -> pickle:
    """Load a pickle file"""
    try:
        with open(file_path,'rb') as file:
            model = pickle.load(file=file)
        logger.debug("Successfully loaded the model")
        return model
    except FileNotFoundError as e:
        logger.error("Couldnt load the model %s", e)
        raise

def load_data(file_path: str)-> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        logger.debug("Data loaded from %s", file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error("File not found error %s",e)
    except Exception as e:
        logger.error("Unexpected error while loading the file %s", e)
        raise


def eval_model(clf, xtest: np.array, ytest: np.array ) -> tuple[dict, np.ndarray]:
    try:
        ypred = clf.predict(xtest)
        accuracy = accuracy_score(ytest, ypred)
        precision = precision_score(ytest,ypred)
        recall = recall_score(ytest,ypred)
        auc = roc_auc_score(ytest,ypred)

        metrics_dict = {
            "accuracy": accuracy,
            "precision": precision,
            "recall":recall,
            'auc':auc
        }

        logger.debug("Model eval metrics calculated")
        return metrics_dict, ypred
    except Exception as e:
        logger.error("Error during model eval")
        raise

def save_metrics(metrics: dict, file_path: str)-> None:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as file:
            json.dump(metrics, file, indent=4)
        logger.debug("Metrics saved to %s", file_path)

    except Exception as e:
        logger.error('Error occurred while saving the metrics: %s', e)
        raise

def main():
    try:
        params = load_yaml('params.yaml')
        clf = load_model("./models/model.pkl")
        test_data = load_data("./data/processed/test_tfidf.csv")

        xtest = test_data.iloc[:, :-1].values
        ytest = test_data.iloc[:,-1].values

        metrics, ypred = eval_model(clf=clf, xtest=xtest, ytest=ytest) 

        with Live(save_dvc_exp=True) as live:
            live.log_metric("accuracy", accuracy_score(ytest, ypred))
            live.log_metric("precision", precision_score(ytest, ypred))
            live.log_metric("recall", recall_score(ytest, ypred))
            live.log_metric("auc", roc_auc_score(ytest,ypred))

            live.log_params(params)

        save_metrics(metrics=metrics, file_path="reports/metrics.json")
    except Exception as e:
        logger.error("Failed to complete the eval process %s", e)
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

