from pydantic import BaseModel, field_validator
import logging
from bs4 import BeautifulSoup
import tiktoken
from typing import Any
import orjson
import pydantic
from bullet.models.utils import validate_prompt


class ZeroShotPrompt(BaseModel):
    prompt: str = """
        Please correctly classify each and every movie review below into negative ("NEG") or 
        positive ("POS") sentiment.
        For example, given the JSON objects below:
        {{"id": "123", "review": "I didn't like this movie"}}
        The correct answer in JSON format would be:
        {{"id": "123", "label": "NEG"}}
        Taking the instructions above into account, generate the correct sentiment label for
        the movie review below:
        {examples}
        Make sure to include both columns in your answers,
        and return the exact same number of outputs as the number of movie reviews.
        Don't include the examples or any commentary.
    """

    examples: str
    preprocess: bool = False
    embeddings_for_model: str = "gpt-3.5-turbo-instruct"
    encoding: Any = None

    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        self.encoding = tiktoken.encoding_for_model(self.embeddings_for_model)
        logging.info(self.examples)

    def __str__(self):
        output = self.prompt.format(examples=self.examples_dict)
        logging.info(f"Prompt output: {output}")
        return output

    def __len__(self):
        return len(self.encoding.encode(str(self)))
    
    @pydantic.computed_field()
    @property
    def examples_dict(self) -> dict:

        try:
            logging.info(f"Before: {self.examples}")
            processed = BeautifulSoup(markup = self.examples, features = "lxml").text
            response_dict = orjson.loads(processed)
            logging.info(f"After validation: {response_dict}")
            return response_dict

        except orjson.JSONDecodeError as error:
            logging.error("Error validating response")
            logging.error("Trying to reformat response...")
            
            response = validate_prompt(
                string = self.examples,
                error = error
            )
            logging.info(f"Fixed prompt: {response}")
            response_dict = orjson.loads(response)
            return response_dict

