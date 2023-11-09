import aiohttp
import asyncio
import openai
import pydantic
from openai import OpenAI
import os
import logging
from pydantic import BaseModel
from pydantic_core import ValidationError
from typing import List, Any
import tenacity
import tiktoken
from tqdm.asyncio import tqdm
import pandas as pd
import time
import random
from httpx import ReadTimeout

from bullet.models.prompts.classification import ZeroShotPrompt
from bullet.core.utils import create_prompt_batches
from bullet.core.postprocess import clean_string
from bullet.models.responses.classification import ClassificationResponse

class SentimentClassifier(BaseModel):
    provider: str = "openai"
    model: str = "gpt-3.5-turbo-instruct"
    api_key: str = ""
    context: str = """
        You are a helpful AI assistant, skilled in classifying passages 
        of text into Positive or Negative sentiment.
    """
    encoding: Any = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        if self.api_key == "":
            self.api_key = os.environ["OPENAI_API_KEY"]
        
        self.encoding = tiktoken.encoding_for_model(self.model)
        if self.provider != "openai" or self.model != "gpt-3.5-turbo-instruct":
            raise NotImplementedError(
                "Bullet only supports gpt-3.5-turbo-instruct at the moment"
            )
        
    @pydantic.computed_field()
    @property
    def api(self) -> int:
        client = OpenAI(api_key = self.api_key)
        client.timeout = 2

        return client


    @tenacity.retry(
        retry=tenacity.retry_if_exception_type(
            exception_types = (
                openai._exceptions.RateLimitError,
                openai._exceptions.APIConnectionError,
                ReadTimeout
            )
        ),
        wait=tenacity.wait_random(min=2, max=10),
        stop=tenacity.stop_after_attempt(5),
        reraise=True,
    )
    def predict_pandas(
        self,
        df: pd.DataFrame,
        temperature: float = 0.0,
        top_p: float = 0.9,
        n: int = 1,
        max_tokens: int = 20
    ):
        
        results = []
        for _, row in df.iterrows():
            prompt = str(ZeroShotPrompt(review = row["text"]))
            
            response = self.api.completions.create(
                model=self.model,
                prompt=str(prompt),
                temperature=temperature,
                n=n,
                max_tokens=max_tokens,
            )

            logging.info(f"OpenAI Response: {response}")
            output = response.choices[0].text
            logging.info(type(output))
            response = ClassificationResponse(response = output)
            results.append(response)

        return results


    @tenacity.retry(
        retry=tenacity.retry_if_exception_type(
            exception_types = (
                openai._exceptions.RateLimitError,
                openai._exceptions.APIConnectionError,
                ReadTimeout
            )
        ),
        wait=tenacity.wait_random(min=2, max=10),
        stop=tenacity.stop_after_attempt(5),
        reraise=True,
    )
    def predict(
        self,
        text: List[dict],
        temperature: float = 0.1,
        top_p: float = 0.9,
        n: int = 1,
        max_tokens = 100
    ) -> List[ClassificationResponse]:
        logging.info(f"Input text: {text}")
        logging.info(f"Temperature: {temperature}")
        logging.info(f"top_p: {top_p}")
        logging.info(f"n: {n}")

        responses = []

        for review in text:
            logging.info(f"Text: {review}")
            prompt = ZeroShotPrompt(review=review)
            logging.info(f"Prompt: {str(prompt)}")
            batch_n_tokens = len(self.encoding.encode(str(prompt)))
            logging.info(f"Number of tokens for batch: {batch_n_tokens}")

            response = self.api.completions.create(
                model=self.model,
                prompt=str(prompt),
                temperature=temperature,
                n=n,
                max_tokens=max_tokens
            )

            logging.info(f"OpenAI Response: {response}")
            output = response.choices[0].text
            output = ClassificationResponse(response = output)
            responses.append(output)

        logging.info(f"Results: {responses}")

        return responses