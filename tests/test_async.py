from src.bullet.core.sentiment import SentimentClassifier
import logging
from tests.fixtures import environment
from tests.fixtures.reviews import text, results
import logging
import pytest

logging.basicConfig(level = "ERROR")

@pytest.mark.asyncio
async def test_classify_list(text, results):

    classifier = SentimentClassifier()
    result = await classifier.apredict(
        text = text * 10
    )

    logging.info(f"Result: {result}")
    assert len(result) == len(results * 10)
    for item in result:
        logging.error(f"Item: {item}")
        assert item.label
