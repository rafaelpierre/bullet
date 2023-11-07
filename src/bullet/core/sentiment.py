import aiohttp
import asyncio
import openai
import os
import logging
from pydantic import BaseModel
from pydantic_core import ValidationError
from typing import List, Any
import tenacity
import tiktoken
from tqdm.asyncio import tqdm
import pandas as pd

from bullet.models.prompts.classification import ZeroShotPrompt
from bullet.core.utils import create_prompt_batches
from bullet.core.postprocess import clean_string
from bullet.models.responses.classification import ClassificationResponse


OPENAI_MAX_TOKEN_SIZE = 4097


class SentimentClassifier(BaseModel):
    provider: str = "openai"
    model: str = "gpt-3.5-turbo-1106"
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
            openai.api_key = self.api_key
            self.encoding = tiktoken.encoding_for_model(self.model)
        if self.provider != "openai" or self.model != "gpt-3.5-turbo-1106":
            raise NotImplementedError(
                "Bullet only supports gpt-3.5-turbo-instruct at the moment"
            )

    @tenacity.retry(
        retry=tenacity.retry_if_exception_type(
            exception_types = (openai.error.RateLimitError)
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
        batch: bool = False,
        max_batch_size: int = 500,
        parse_json: bool = False,
        max_tokens: int = 20
    ):
        
        results = []
        for _, row in df.iterrows():
            payload = f'\"id\": {row["id"]}, "text": {row["text"]}'
            prompt = ZeroShotPrompt(examples = payload)
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": str(prompt)
                }
            ]
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                n=n,
                max_tokens=max_tokens,
                response_format={ "type": "json_object" }
            )

            logging.info(f"OpenAI Response: {response}")
            output = response.choices[0]["message"]["content"]
            logging.info(type(output))
            response = ClassificationResponse(response = output)
            results.append(response)

        return results


    @tenacity.retry(
        retry=tenacity.retry_if_exception_type(
            exception_types = (openai.error.RateLimitError)
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
        batch: bool = False,
        max_batch_size: int = 500,
        parse_json: bool = False,
    ) -> List[ClassificationResponse]:
        logging.info(f"Input text: {text}")
        logging.info(f"Temperature: {temperature}")
        logging.info(f"top_p: {top_p}")
        logging.info(f"n: {n}")

        responses = []

        if batch:
            batches = create_prompt_batches(
                examples=text, max_batch_size=max_batch_size
            )
            logging.info(f"Number of batches: {len(batches)}")
        else:
            batches = text

        for batch in batches:
            logging.info(f"Batch: {batch}")
            prompt = ZeroShotPrompt(examples=batch)
            logging.info(f"Prompt: {str(prompt)}")
            batch_n_tokens = len(self.encoding.encode(str(prompt)))
            logging.info(f"Number of tokens for batch: {batch_n_tokens}")
            len_prompt = len(prompt)

            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": str(prompt)
                }
            ]

            response = openai.ChatCompletion.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                n=n,
                max_tokens=100,
                response_format={ "type": "json_object" }
            )

            logging.info(f"OpenAI Response: {response}")
            output = response.choices[0]["message"]["content"]
            output = ClassificationResponse(response = output)
            responses.append(output)

        logging.info(f"Results: {responses}")

        return responses

    @tenacity.retry(
        retry=tenacity.retry_if_exception_type(
            exception_types = (openai.error.RateLimitError)
        ),
        wait=tenacity.wait_random(min=0.5, max=2),
        stop=tenacity.stop_after_attempt(5),
        reraise=True,
    )
    async def dispatch(
        self,
        prompt: ZeroShotPrompt,
        temperature: float = 0.1,
        top_p: float = 0.9,
        n: int = 1,
        parse_json: bool = False,
        max_tokens: int = 20
    ):
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": str(prompt)
            }
        ]

        result = openai.ChatCompletion.acreate(
            model=self.model,
            messages=messages,
            temperature=temperature,
            n=n,
            max_tokens=max_tokens,
            response_format={ "type": "json_object" }
        )

        output = await result
        output = ClassificationResponse(response = output.choices[0]["message"]["content"])

        return output

    async def apredict(
        self,
        text: List[str],
        temperature: float = 0.1,
        top_p: float = 0.9,
        n: int = 1,
        parse_json: bool = False,
        parallelism: int = 10,
    ) -> dict:
        logging.info(f"Input text: {text}")
        logging.info(f"Temperature: {temperature}")
        logging.info(f"top_p: {top_p}")
        logging.info(f"n: {n}")

        openai.aiosession.set(aiohttp.ClientSession())

        pending_requests = len(text)
        results = []
        chunks = [
            text[
                i * parallelism : i * parallelism
                + min(parallelism, pending_requests - i * parallelism)
            ]
            for i in range(pending_requests // parallelism)
        ]

        for chunk in tqdm(chunks):
            prompts = [ZeroShotPrompt(examples=example) for example in chunk]

            async_responses = [
                self.dispatch(
                    prompt=prompt,
                    temperature=temperature,
                    top_p=top_p,
                    n=n,
                    parse_json=parse_json,
                )
                for prompt in prompts
            ]
            results.extend(await asyncio.gather(*async_responses))
            pending_requests -= parallelism

        await openai.aiosession.get().close()

        return results
