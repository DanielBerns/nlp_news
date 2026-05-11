import yaml
from pydantic import BaseModel
from sqlmodel import create_engine, SQLModel
from pathlib import Path

import logging
import sys

def setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

class Config(BaseModel):
    input_dir: str
    db_path: str = "nlp_news.db"
    language: str = "en"  # "en" or "es"
    n_gram_size: int = 2
    min_support: float = 0.1
    min_confidence: float = 0.5

SPACY_MODELS = {
    "en": "en_core_web_sm",
    "es": "es_core_news_sm"
}

def load_config(config_path: str = "config.yaml") -> Config:
    with open(config_path, "r") as f:
        data = yaml.safe_load(f)
    return Config(**data)

def get_engine(db_path: str):
    sqlite_url = f"sqlite:///{db_path}"
    return create_engine(sqlite_url)

def load_spacy_model(lang: str):
    import spacy
    import subprocess
    model_name = SPACY_MODELS.get(lang, "en_core_web_sm")
    try:
        return spacy.load(model_name)
    except OSError:
        print(f"Downloading spacy model {model_name}...")
        model_url = f"https://github.com/explosion/spacy-models/releases/download/{model_name}-3.8.0/{model_name}-3.8.0-py3-none-any.whl"
        subprocess.run(["uv", "pip", "install", model_url], check=True)
        return spacy.load(model_name)

def init_db(db_path: str):
    from nlp_news.models import Document, NGram, DocumentNGram, Itemset, AssociationRule
    engine = get_engine(db_path)
    SQLModel.metadata.create_all(engine)
