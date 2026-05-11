from typing import List, Optional
from sqlmodel import Field, SQLModel, Relationship, Column, JSON

from datetime import datetime, timezone

class Document(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    path: str = Field(index=True, unique=True)
    content: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source_url: Optional[str] = Field(default=None)

class NGram(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    text: str = Field(index=True, unique=True)
    n: int = Field(index=True)

class DocumentNGram(SQLModel, table=True):
    document_id: int = Field(foreign_key="document.id", primary_key=True)
    ngram_id: int = Field(foreign_key="ngram.id", primary_key=True)
    count: int = Field(default=1)

class Itemset(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    document_id: int = Field(foreign_key="document.id", index=True)
    items: List[int] = Field(sa_column=Column(JSON))

class AssociationRule(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    antecedents: List[str] = Field(sa_column=Column(JSON))
    consequents: List[str] = Field(sa_column=Column(JSON))
    support: float
    confidence: float
    lift: float
