from pydantic import BaseModel, ConfigDict, field_validator, model_validator
from bullet.core.postprocess import clean_string
from bullet.models.utils import validate_response
import logging
import orjson
import os
import openai
import pydantic

class ClassificationResponse(BaseModel):

    response: str

    @pydantic.computed_field()
    @property
    def response_dict(self) -> dict:
        try:
            logging.info(f"Response: {self.response}")
            logging.info(f"Type: {type(self.response)}")
            clean = (
                self.response
                .replace("\n", "")
                .replace("[", "")
                .replace("]", "")
                .strip()
            )
            logging.info(f"Clean: {clean}")
            response_dict = orjson.loads(clean)
            return response_dict
        
        except orjson.JSONDecodeError as error:
            logging.error("Error validating response")
            logging.error("Trying to reformat response...")
            
            response = validate_response(
                string = self.response,
                error = error
            )
            logging.info(f"Fixed response: {response}")
            response_dict = orjson.loads(response)
            
            return response_dict
        
    @pydantic.computed_field()
    @property
    def id(self) -> int:
        return int(self.response_dict["id"])
    
    @pydantic.computed_field()
    @property
    def label(self) -> str:
        return self.response_dict["label"]
    

            



