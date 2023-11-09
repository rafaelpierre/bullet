from src.bullet.core.sentiment import SentimentClassifier
from src.bullet.models.responses.classification import ClassificationResponse
import logging
from tests.fixtures import environment
from tests.fixtures.reviews import text, results, df
import logging
import pandas as pd

logging.basicConfig(level = "ERROR")

def test_classify_list(text, results):

    classifier = SentimentClassifier()
    result = classifier.predict(
        text = text
    )

    logging.info(f"Result: {result}")
    assert len(result) == len(results)
    for item in result:
        logging.error(f"Item: {item}")
        assert item.response

def test_predict_pandas(df):

    logging.info(f"Input DF: {df}")
    classifier = SentimentClassifier()
    results = classifier.predict_pandas(df)

    assert results

def test_few_shot(df):

    logging.info(f"Input DF: {df}")
    classifier = SentimentClassifier()
    results = classifier.predict_pandas(df)

    assert results