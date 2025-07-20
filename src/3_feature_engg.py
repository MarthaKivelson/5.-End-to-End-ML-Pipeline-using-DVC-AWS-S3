import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
import yaml

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('feature-ENGG')
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

log_file_path = os.path.join(log_dir, "feature_engg.log")
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


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from csv file"""
    try:
        df = pd.read_csv(file_path)
        df.fillna("", inplace=True)
        logger.debug("Data loaded successfully")
        return df
    except pd.errors.ParserError as e:
        logger.error("Failed to parse the csv file %s", e)
        raise
    except Exception as e:
        logger.error("Failed to load the data %s", e)
        raise


def apply_tfidf(train_data:pd.DataFrame, test_data:pd.DataFrame, max_features: int) -> tuple:
    try:
        vec = TfidfVectorizer(max_features=max_features)

        xtrain = train_data['text'].values
        ytrain = train_data['target'].values
        xtext = test_data['text'].values
        ytest = test_data['target'].values

        xtrain_vec = vec.fit_transform(xtrain)
        xtest_vec = vec.fit_transform(xtext)

        train_df = pd.DataFrame(xtrain_vec.toarray())
        train_df['label'] = ytrain

        test_df = pd.DataFrame(xtest_vec.toarray())
        test_df['label'] = ytest

        logger.debug("TfIDF applied")
        return train_df, test_df
    except Exception as e:
        logger.error("Error during TfIDF transformation: %s", e)
        raise

def sav_data(df: pd.DataFrame, file_path: str) -> None:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=True)
        logger.debug("Data saved to %s", file_path)
    except Exception as e:
        logger.error("Unexpected error occured %s",e)
        raise

def main():
    try:
        max_features = load_yaml('params.yaml')["3_feature_engg"]["max_features"]
        
        train_data = load_data("./data/interim/train_processed.csv")
        test_data = load_data("./data/interim/test_processed.csv")

        train_df, test_df = apply_tfidf(train_data=train_data, test_data=test_data, max_features=max_features)
        
        sav_data(train_df,os.path.join("./data","processed","train_tfidf.csv"))
        sav_data(train_df,os.path.join("./data","processed","test_tfidf.csv"))
    except Exception as e:
        logger.error("Failed to complete the feature engg process %s", e)
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
