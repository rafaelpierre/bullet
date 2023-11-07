import orjson
import logging
from bs4 import BeautifulSoup
import ast
import re

def parse_json(batches: str) -> dict:
    logging.info(f"Batches to be parsed: {batches}")

    soup = BeautifulSoup(batches, "lxml")
    parsed = orjson.loads(orjson.dumps(soup.text))

    return ast.literal_eval(parsed)


def clean_string(batches: str) -> dict:

    logging.info(f"Batches to be parsed: {batches}")

    id = re.findall(pattern="([0-9]+)", string = batches)[0]
    label = re.findall(pattern="(NEG|POS)", string = batches)[0]

    return id, label
