import pytest 
import os

@pytest.fixture(scope = "session")
def api_key():
    return os.environ["OPENAI_API_KEY"]