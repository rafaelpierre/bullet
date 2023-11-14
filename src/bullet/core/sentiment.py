"""
Main module for Sentiment Classification Functionality.

Contains the SentimentClassifier class, which ships with all functions to perform Zero Shot
and Few Shot sentiment classification.
"""

import openai
import pydantic
from openai import OpenAI
import os
import logging
from pydantic import BaseModel
from typing import List, Any
import tenacity
import tiktoken
import tqdm
import pandas as pd
from httpx import ReadTimeout

from bullet.models.prompts.classification import ZeroShotPrompt, FewShotPrompt
from bullet.models.responses.classification import PromptResponse, ClassificationResponse


class SentimentClassifier(BaseModel):
    """
        SentimentClassifier contains methods for performing Zero-Shot and Few-Shot classification.
    """
    provider: str = "openai"
    """Large Language Model (LLM) provider. Currently supported values: openai"""

    model: str = "gpt-3.5-turbo-instruct"
    """Large Language Model (LLM) identification. Currently supported values: gpt-3.5-turbo-instruct"""

    api_key: str = ""
    """API key for your LLM. Can also be set through environment variables, e.g. OPENAI_API_KEY."""
    
    context: str = """
        You are a helpful AI assistant, skilled in classifying passages 
        of text into Positive or Negative sentiment.
    """
    """Initial prompt context."""

    encoding: Any = None
    """Placeholder for embeddings object."""

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
        """Instantiates and returns an OpenAI client object."""
        client = OpenAI(api_key=self.api_key)
        client.timeout = 2

        return client

    @tenacity.retry(
        retry=tenacity.retry_if_exception_type(
            exception_types=(
                openai._exceptions.RateLimitError,
                openai._exceptions.APIConnectionError,
                ReadTimeout,
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
        max_tokens: int = 20,
    ) -> pd.DataFrame:
        """
        Runs sentiment classification on top of a Pandas Dataframe.
        Parameters:
        :param df: Pandas Dataframe. Must contain a 'text' column with the text to be classified.
        :param temperature: Generation temperature. Higher values will result in less deterministic
        :param output (recommended to use 0.0)
        :param n: Number of generated outputs. Recommended: 1
        :param max_tokens: maximum number of generated tokens. Defaults to 20.
        :returns: Pandas DataFrame containing prediction results.
        """
        results = []
        for _, row in df.iterrows():
            prompt = str(ZeroShotPrompt(review=row["text"]))

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
            response = PromptResponse(response=output)
            results.append(response)

        return ClassificationResponse(results = results)

    @tenacity.retry(
        retry=tenacity.retry_if_exception_type(
            exception_types=(
                openai._exceptions.RateLimitError,
                openai._exceptions.APIConnectionError,
                ReadTimeout,
            )
        ),
        wait=tenacity.wait_random(min=2, max=10),
        stop=tenacity.stop_after_attempt(5),
        reraise=True,
    )
    def predict(
        self,
        text: List[str],
        temperature: float = 0.1,
        top_p: float = 0.9,
        n: int = 1,
        max_tokens=100,
    ) -> ClassificationResponse:
        """
        Runs sentiment classification on top of list of strings.
        Parameters:
        :param text: List of strings to be classified.
        :param temperature: Generation temperature. Higher values will result in less deterministic
        output (recommended to use 0.0)
        :param n: Number of generated outputs. Recommended: 1
        :param max_tokens: maximum number of generated tokens. Defaults to 20.
        :returns: ClassificationResponse object.
        """
        logging.info(f"Input text: {text}")
        logging.info(f"Temperature: {temperature}")
        logging.info(f"top_p: {top_p}")
        logging.info(f"n: {n}")

        responses = []

        for review in tqdm.tqdm(text):
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
                max_tokens=max_tokens,
            )

            logging.info(f"OpenAI Response: {response}")
            output = response.choices[0].text
            output = PromptResponse(response=output)
            responses.append(output)

        logging.info(f"Results: {responses}")

        return ClassificationResponse(results = responses)

    @tenacity.retry(
        retry=tenacity.retry_if_exception_type(
            exception_types=(
                openai._exceptions.RateLimitError,
                openai._exceptions.APIConnectionError,
                ReadTimeout,
            )
        ),
        wait=tenacity.wait_random(min=2, max=10),
        stop=tenacity.stop_after_attempt(5),
        reraise=True,
    )
    def predict_few_shot(
        self,
        reviews: List[str],
        examples: List[str],
        temperature: float = 0.1,
        top_p: float = 0.9,
        n: int = 1,
        max_tokens=100,
    ) -> ClassificationResponse:
        """Runs sentiment classification on top of list of strings using Few-Shot Examples.
        
        Params:
        reviews(List[str]): List of strings to be classified.
        examples(List[str]): List of strings containing few-shot examples.
        temperature(float): Generation temperature. Higher values will result in less deterministic outputs.
        output (recommended to use 0.0)
        top_p(float): Top probability value for output selection.
        n(int): Number of generated outputs. Recommended: 1
        max_tokens(int): maximum number of generated tokens. Defaults to 20.
        """
        logging.info(f"Input text: {reviews}")
        logging.info(f"Temperature: {temperature}")
        logging.info(f"top_p: {top_p}")
        logging.info(f"n: {n}")

        results = []
        for review in reviews:
            logging.info(f"Text: {review}")
            prompt = FewShotPrompt(review=review, examples=examples)
            logging.info(f"Prompt: {str(prompt)}")

            response = self.api.completions.create(
                model=self.model,
                prompt=str(prompt),
                temperature=temperature,
                n=n,
                max_tokens=max_tokens,
            )

            logging.info(f"OpenAI Response: {response}")
            output = response.choices[0].text
            output = PromptResponse(response=output)

            logging.info(f"Result: {output}")
            results.append(output)

        return ClassificationResponse(results = results)
