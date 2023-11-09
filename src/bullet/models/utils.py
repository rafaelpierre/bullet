import openai
from typing import List
import tenacity
import orjson
import os
from openai import OpenAI

VALIDATION_MODEL = "gpt-3.5-turbo-1106"

@tenacity.retry(
    retry=tenacity.retry_if_exception_type(
        exception_types = (
            openai._exceptions.RateLimitError,
            openai._exceptions.APIConnectionError,
            orjson.JSONDecodeError
        )
    ),
    wait=tenacity.wait_random(min=0.5, max=1),
    stop=tenacity.stop_after_attempt(5),
    reraise=False,
)
def validate_prompt(
    string: str,
    error: Exception,
    target_format: str = "json",
    stop = None,
    temperature: float = 0.0,
    n: int = 1,
    max_tokens: int = 20
) -> str:

    api = OpenAI(api_key = os.environ["OPENAI_API_KEY"])
    placeholder = """
        You are an expert in Python.
        I'm trying to parse the following string into a {target_format}: "{string}"
        However when I try to do so, I get the following error: {error}
        Can you please fix the error on this string, so that I can properly parse it
        as a {target_format}?
        Don't include any commentary, only the fixed string.
    """

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": placeholder.format(
                string = string,
                target_format = target_format,
                error = str(error),
            ),
        }
    ]
    response = api.chat.completions.create(
        model = VALIDATION_MODEL,
        messages = messages,
        temperature = temperature,
        n = n,
        max_tokens = max_tokens,
        stop = stop,
        response_format = { "type": "json_object" }
    )

    return response.choices[0].message.content



@tenacity.retry(
    retry=tenacity.retry_if_exception_type(
        exception_types = (
            openai._exceptions.RateLimitError,
            openai._exceptions.APIConnectionError,
            orjson.JSONDecodeError
        )
    ),
    wait=tenacity.wait_random(min=0.5, max=1),
    stop=tenacity.stop_after_attempt(5),
    reraise=False,
)
def validate_response(
    string: str,
    error: Exception,
    target_format: str = "json",
    stop = None,
    temperature: float = 0.0,
    n: int = 1,
    max_tokens: int = 20
) -> str:

    api = OpenAI(api_key = os.environ["OPENAI_API_KEY"])
    placeholder = """
        You are an expert in Python.
        I'm trying to parse the following string into a {target_format}: "{string}"
        However when I try to do so, I get the following error: {error}
        Can you please fix the error on this string, so that I can properly parse it
        as a {target_format}?
        Don't include any commentary, only the fixed string.
    """

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": placeholder.format(
                string = string,
                target_format = target_format,
                error = str(error),
            ),
        }
    ]

    response = api.chat.completions.create(
        model = VALIDATION_MODEL,
        messages = messages,
        temperature = temperature,
        n = n,
        max_tokens = max_tokens,
        stop = stop,
        response_format = { "type": "json_object" }
    )

    return response.choices[0].message.content



