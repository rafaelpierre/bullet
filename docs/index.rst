🚅 bullet: A Zero-Shot / Few-Shot Learning, LLM Based, text classification framework
=====================================================================================

Motivation
----------

* Besides the fact that LLMs have a huge power in **generative** use cases, there is a use case that is quite frequently overlooked by frameworks such as [LangChain](https://www.langchain.com/): **Text Classification**.
* 🚅 **bullet** was created to address this. It leverages the power of **ChatGPT**, while removing any boilerplate code that is needed for performing **text classification** using either **Zero Shot** or **Few Shot Learning**.

## Getting Started

1. Install `bullet`: `pip install git+https://github.com/rafaelpierre/bullet`
2. Configure your `OPENAI_API_KEY`
3. You should be good to go

### Zero-Shot Classification

```python
from bullet.core.sentiment import SentimentClassifier


df_train_sample = df_train.sample(n = 50)

classifier = SentimentClassifier()
result = classifier.predict_pandas(df_train_sample)
```

### Few-Shot Classification

```python

# Define Few Shot examples

template = "Review: \"{review}\"\nLabel: \"{label}\""
examples = [
    template.format(
        review = row["text"],
        label = "POS" if row["label"] == 1 else "NEG"
    )
    for _, row
    in df_train.sample(3).iterrows()
]

df_test_sample = dataset["test"].to_pandas().sample(100)
reviews = df_test_sample.text.values

results = classifier.predict_few_shot(
    reviews = reviews,
    examples = examples
)
```


API Reference
-------------

If you are looking for information on a specific function, class, or
method, this part of the documentation is for you.

.. toctree::
   :maxdepth: 2

   api