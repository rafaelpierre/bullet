from src.bullet.models.prompts.classification import ZeroShotPrompt
from src.bullet.models.responses.classification import ClassificationResponse
from dotenv import load_dotenv
import pytest

load_dotenv()

def test_create():

    prompt = ZeroShotPrompt(
        review = "This is a review"
    )

    assert prompt
    assert prompt.review

def test_create_error():

    with pytest.raises(Exception):
        prompt = ZeroShotPrompt(
            review = ["test"]
        )

def test_response():

    response = ClassificationResponse(response = "NEG")

    assert response
    assert response.response
