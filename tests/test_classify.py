from src.bullet.core.sentiment import SentimentClassifier
from src.bullet.models.responses.classification import ClassificationResponse
import logging
from tests.fixtures import environment
from tests.fixtures.api_key import api_key
from tests.fixtures.reviews import (
    text,
    results,
    df_train,
    df_test
)
import os

import logging
import pandas as pd

logging.basicConfig(level = "ERROR")

def test_classify_list(text, results, api_key):

    classifier = SentimentClassifier(api_key = api_key)
    result = classifier.predict(
        text = text
    )

    logging.info(f"Result: {result}")
    assert len(result) == len(results)

def test_predict_pandas(df_train, api_key):

    logging.info(f"Input DF: {df_train}")
    classifier = SentimentClassifier(api_key = api_key)
    results = classifier.predict_pandas(df_train)
    logging.info(f"Predict pandas - results: {results}")

    assert results

def test_few_shot(df_train, df_test, api_key):

    logging.info(f"Input DF - examples: {df_train}")
    logging.info(f"Input DF - test set: {df_test}")
    classifier = SentimentClassifier(api_key = api_key)
    results = classifier.predict_few_shot(
        reviews = df_test.text.values,
        examples = df_train.sample(3).text.values
    )

    logging.info(f"Predict few shot - results: {results}")
    assert results