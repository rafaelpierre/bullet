import pytest
import pandas as pd
from datasets import load_dataset


@pytest.fixture()
def df():

    dataset = load_dataset("imdb", split = "train")
    return dataset.to_pandas().sample(10)

@pytest.fixture()
def text(df):

    return df.text.values

@pytest.fixture()
def results(df):

    return df.label.values