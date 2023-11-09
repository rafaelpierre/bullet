import pytest
import pandas as pd
from datasets import load_dataset


@pytest.fixture(scope = "session")
def df_train():

    dataset = load_dataset("imdb", split = "train")
    return dataset.to_pandas().sample(10)

@pytest.fixture(scope = "session")
def df_test():

    dataset = load_dataset("imdb", split = "test")
    return dataset.to_pandas().sample(10)

@pytest.fixture(scope = "session")
def text(df_train):

    return df_train.text.values

@pytest.fixture(scope = "session")
def results(df_train):

    return df_train.label.values

@pytest.fixture()
def text_test(df_test):

    return df_test.text.values

@pytest.fixture()
def results_test(df_test):

    return df_test.label.values

