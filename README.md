# 🚅 bullet: A Zero-Shot / Few-Shot Learning, LLM Based, text classification framework

![version](https://img.shields.io/badge/version-0.0.1-red?style=for-the-badge) ![ChatGPT](https://img.shields.io/badge/chatGPT-74aa9c?style=for-the-badge&logo=openai&logoColor=white) ![python](https://img.shields.io/badge/python-3.11-blue?style=for-the-badge) ![build](https://img.shields.io/badge/coverage-92%25-green?style=for-the-badge) ![coverage](https://img.shields.io/badge/coverage-96%25-green?style=for-the-badge)

## Motivation

* Besides the fact that LLMs have a huge power in **generative** use cases, there is a use case that is quite frequently overlooked by frameworks such as LangChain: **Text Classification**.
* 🚅 **bullet** was created to address this. It leverages the power of **ChatGPT**, while removing any boilerplate code that is needed for performing **text classification** using either **Zero Shot** or **Few Shot Learning**.

## Getting Started

1. Install `bullet`: `pip install git+https://github.com/rafaelpierre/bullet`
2. Configure your `OPENAI_API_KEY`
3. You should be good to go:

```python
from bullet.core.sentiment import SentimentClassifier


df_train_sample = df_train.sample(n = 50)

classifier = SentimentClassifier()
result = classifier.predict_pandas(df_train_sample)
```

Full working example on a Jupyter Notebook can be found in `notebooks/sandbox.ipynb`

### Development

* From a terminal window, start a Python virtual environment and activate it:

```bash
python -m venv .venv
source .venv/bin/activate
```

* Install `tox`:

```bash
pip install tox
```

* Running unit tests:

```bash
tox .
```
