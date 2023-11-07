from setuptools import find_packages, setup
import bullet
from pathlib import Path

PACKAGE_REQUIREMENTS = [
    "openai",
    "pydantic",
    "tqdm",
    "tenacity",
    "tiktoken",
    "python-rapidjson",
    "beautifulsoup4",
    "lxml"
]

current_dir = Path(__file__).parent.parent
long_description = (current_dir / "README.md").read_text()

setup(
    name="bullet",
    packages=find_packages(),
    setup_requires=["setuptools", "wheel"],
    install_requires=PACKAGE_REQUIREMENTS,
    entry_points={"console_scripts": ["bullet = bullet.client.entrypoint:main"]},
    version=bullet.__version__,
    long_description=long_description,
    long_description_content_type="text/markdown",
    authors=["Rafael Pierre"],
)