# Importing necessary libraries
import os
import logging
from sklearn.preprocessing import LabelEncoder
from nltk.stem.porter import PorterStemmer
import string
import numpy as np        # For numerical operations
import pandas as pd       # For data manipulation and analysis
import matplotlib.pyplot as plt  # For data visualization
import nltk
from nltk.corpus import stopwords    # For stopwords


# Downloading NLTK data
nltk.download('stopwords')   # Downloading stopwords data
nltk.download('punkt')       # Downloading tokenizer data

log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('data_preprocessing')
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

log_file_path = os.path.join(log_dir, "data_preprocessing.log")
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel("DEBUG")

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

def transform_text(text: str):
    """
    Transform text by lowering, tokenizing, stemming, removing special char stopwords & punctuations
    """
    text = text.lower()
    text = nltk.word_tokenize(text)
    ps = PorterStemmer()

    y = []

    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

def preprocess_df(df, text_column="text", target_column="target") -> pd.DataFrame:
    try:
        logger.debug("Preprocessing started")
        encoder = LabelEncoder()
        df["target"] = encoder.fit_transform(df["target"])
        logger.debug("Encoding done")

        df = df.drop_duplicates(keep="first")
        logger.debug("Dropped duplicates")

        df.loc[:, text_column] = df[text_column].apply(transform_text)
        logger.debug("Text column transformed")
        return df
    except KeyError as e:
        logger.error("Column not found %s", e)
    except Exception as e:
        logger.error("Error during text normalization %s", e)
        raise

def main():
    try:
        train_data = pd.read_csv("./data/raw/train.csv")
        test_data = pd.read_csv("./data/raw/test.csv")
        logger.debug("Data loaded properly")

        train_data_processed = preprocess_df(train_data)
        test_data_preprocessed = preprocess_df(test_data)

        data_path = os.path.join("./data", "interim")
        os.makedirs(data_path, exist_ok=True)

        train_data_processed.to_csv(os.path.join(data_path,"train_processed.csv"), index=True)
        test_data_preprocessed.to_csv(os.path.join(data_path, "test_processed.csv"),index=True)
        logger.debug("Processed data saved to %s", data_path)
    except FileNotFoundError as e:
        logger.error("File not found", e)
    except pd.errors.EmptyDataError as e:
        logger.error("No data %s", e)
    except Exception as e:
        logger.error("Failed to complete the data transformation precess %s", e)
        print(f"Error: {e}")

    
if __name__ == "__main__":
    main()
    