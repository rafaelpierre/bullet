from pydantic import BaseModel
import pydantic
from typing import Any, List
import tiktoken
import logging

class PromptResponse(BaseModel):
    response: str
    embeddings_for_model: str = "gpt-3.5-turbo-instruct"
    encoding: Any = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        logging.info(self.response)

    @pydantic.computed_field()
    @property
    def label(self) -> int:
        if "POS" in self.response:
            return 1
        elif "NEG" in self.response:
            return 0
        else:
            raise ValueError(f"Invalid response: {self.response}")

    def __str__(self):
        return self.response

    def __int__(self):
        return int("POS" in self.response)

    def __len__(self):
        encoding = tiktoken.encoding_for_model(self.embeddings_for_model)
        return len(encoding.encode(str(self)))
    
class ClassificationResponse(BaseModel):

    results: List[PromptResponse]

    def to_list(self):
        return [response.label for response in self.results]
    
    def __len__(self):
        return len(self.results)
