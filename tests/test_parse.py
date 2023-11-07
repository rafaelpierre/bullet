from src.bullet.core.postprocess import parse_json, clean_string
import logging

def test_parse_fail():

    example = '\n[{\'sentence\': "I didn\'t like this movie", \'label\': \'NEG\'}]'
    parsed = parse_json(batches = example)
    logging.info(parsed)
    logging.info(f"Type: {type(parsed)}")

    assert parsed
    assert isinstance(parsed, list)
    assert isinstance(parsed[0], dict)

def test_parse_succeed():

    example = """[{"sentence": "I didnt like this movie", "label": "NEG"}]"""
    parsed = parse_json(batches = example)
    logging.info(parsed)
    logging.info(f"Type: {type(parsed)}")

    assert parsed
    assert isinstance(parsed, list)
    assert isinstance(parsed[0], dict)

def test_clean_string():

    test = '"123","NEG"'
    clean = clean_string(test)

    assert clean
    assert len(clean) > 0
    assert clean[1] in ("NEG", "POS")