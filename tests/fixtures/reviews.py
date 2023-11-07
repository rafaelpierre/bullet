import pytest
import pandas as pd

@pytest.fixture()
def text():
    reviews = [
        '{"id": "1", "text": "I didn\'t like this movie"}',
        '{"id": "234", "text": "This movie is good!"}',
        '{"id": "666", "text": "I like this movie"}',
        '{"id": "777", "text": "This is trash"}',
        '{"id": "888", "text": "I didn\'t like this movie"}',
        '{"id": "888", "text": "I didn\'t like this movie"}'
    ]

    return reviews

@pytest.fixture()
def results():

    results = [
        {"id": "1", "label": "NEG"},
        {"id": "2", "label": "POS"},
        {"id": "234", "label": "POS"},
        {"id": "666", "label": "NEG"},
        {"id": "777", "label": "NEG"},
        {"id": "888", "label": "NEG"}
    ]

    return results

@pytest.fixture()
def df():

    reviews = [
        {"id": "1", "text": "I didn\'t like this movie"},
        {"id": "234", "text": "This movie is good!"},
        {"id": "666", "text": "I like this movie"}
    ]

    df_ = pd.DataFrame.from_records(reviews)
    return df_