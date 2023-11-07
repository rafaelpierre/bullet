from src.bullet.models.prompts.classification import ZeroShotPrompt
from src.bullet.models.responses.classification import ClassificationResponse
from dotenv import load_dotenv
import pytest

load_dotenv()

def test_create():

    prompt = ZeroShotPrompt(
        examples = "\"id\": 1, \"text\": \"test\"}"
    )

    assert prompt
    assert prompt.examples

def test_create_error():

    with pytest.raises(Exception):
        prompt = ZeroShotPrompt(
            examples = ["test"]
        )

def test_response():

    response = ClassificationResponse(response = '{"id": 1, "label": "POS"}')

    assert response.id is not None
    assert response.response
    assert response.label is not None

def test_response_validate():

    response = ClassificationResponse(response = '{"id": 1, "label": "POS"')

    assert response.id is not None
    assert response.response
    assert response.label is not None