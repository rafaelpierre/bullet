from pydantic import BaseModel
import logging
import tiktoken
from typing import Any, List
from bullet.models.utils import validate_prompt


class ZeroShotPrompt(BaseModel):
    prompt: str = """
        Please correctly classify each and every movie review below into negative ("NEG") or 
        positive ("POS") sentiment.
        Taking the instructions above into account, generate the correct sentiment label for
        the movie review below:
        {review}
        Don't include the examples or any commentary, just the classification result ("NEG" or "POS")
    """

    review: str
    preprocess: bool = False
    embeddings_for_model: str = "gpt-3.5-turbo-instruct"
    encoding: Any = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.encoding = tiktoken.encoding_for_model(self.embeddings_for_model)
        logging.info(self.review)

    def __str__(self):
        output = self.prompt.format(review=self.review)
        logging.info(f"Prompt output: {output}")
        return output

    def __len__(self):
        return len(self.encoding.encode(str(self)))


class FewShotPrompt(BaseModel):
    prompt: str = """
        Please correctly classify each and every movie review below into negative ("NEG") or 
        positive ("POS") sentiment.
        You can find some classification examples below:
        {examples}
        Taking the instructions above into account, generate the correct sentiment label for
        the movie review below:
        {review}
        Don't include the examples or any commentary.
    """

    examples: List[str]
    review: str
    preprocess: bool = False
    embeddings_for_model: str = "gpt-3.5-turbo-instruct"
    encoding: Any = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.encoding = tiktoken.encoding_for_model(self.embeddings_for_model)
        logging.info(self.review)

    def __str__(self):
        output = self.prompt.format(examples=self.examples, review=self.review)
        logging.info(f"Prompt output: {output}")
        return output

    def __len__(self):
        return len(self.encoding.encode(str(self)))
