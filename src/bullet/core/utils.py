from typing import List
import logging
import tiktoken


def create_prompt_batches(
    examples: List[str], max_batch_size=500, model="gpt-3.5-turbo-instruct"
) -> List[str]:
    max_example_size = max_batch_size - 10
    batches = []
    current_batch = ""
    batch_size = 0
    encoding = tiktoken.encoding_for_model(model)

    for example in examples:
        logging.info(f"Batch: {current_batch}")
        logging.info(f"Example: {example}")

        encoded_example = encoding.encode(example)
        encoded_batch = encoding.encode(current_batch)
        truncated_example = encoding.decode(
            encoded_example[: min(max_example_size, len(encoded_example))]
        )

        if len(encoded_batch) + len(truncated_example) <= max_batch_size:
            current_batch += f"{truncated_example}\n"
            batch_size += len(truncated_example)
        else:
            batches.append(current_batch)
            current_batch = f"{example}\n"
            batch_size = len(truncated_example)

    if current_batch:
        batches.append(current_batch)

    logging.info(f"Batches: {batches}")
    return batches
