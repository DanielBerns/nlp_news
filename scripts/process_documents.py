import os
import spacy
import logging
from datetime import datetime, timezone
from sqlmodel import Session, select, text
from nlp_news.config import load_config, get_engine, init_db, load_spacy_model, setup_logging
from nlp_news.models import Document, NGram, DocumentNGram

logger = logging.getLogger("ingest")

def generate_ngrams(doc, n: int):
    # Filter out punctuation and whitespace
    tokens = [token.text for token in doc if not token.is_punct and not token.is_space]
    
    if n == 1:
        return tokens
    
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngrams.append(" ".join(tokens[i : i + n]))
    return ngrams

def main():
    setup_logging()
    config = load_config()
    logger.info(f"Starting NLP processing for documents in DB: {config.db_path}, Lang: {config.language}")
    
    init_db(config.db_path)
    engine = get_engine(config.db_path)
    
    with Session(engine) as session:
        logger.info("Clearing existing n-gram and rule data (keeping documents)")
        session.execute(text("DELETE FROM documentngram"))
        session.execute(text("DELETE FROM itemset"))
        session.execute(text("DELETE FROM associationrule"))
        session.execute(text("DELETE FROM ngram"))
        session.commit()
    
    nlp = load_spacy_model(config.language)
    logger.info(f"Loaded spaCy model for {config.language}")

    with Session(engine) as session:
        documents = session.exec(select(Document)).all()
        logger.info(f"Found {len(documents)} documents in database to process")
        
        ngram_cache = {}  # text -> id mapping
        
        # Generator for nlp.pipe
        texts = (doc.content.lower() for doc in documents)
        
        for db_doc, spacy_doc in zip(documents, nlp.pipe(texts, batch_size=50)):
            logger.info(f"Processing document ID {db_doc.id}: {db_doc.path[:50]}...")
            
            ngrams_list = generate_ngrams(spacy_doc, config.n_gram_size)
            
            ngram_counts = {}
            for text_val in ngrams_list:
                ngram_counts[text_val] = ngram_counts.get(text_val, 0) + 1
            
            logger.debug(f"Found {len(ngram_counts)} unique {config.n_gram_size}-grams")
            
            new_ngrams = []
            for text_val in ngram_counts.keys():
                if text_val not in ngram_cache:
                    new_ngrams.append(NGram(text=text_val, n=config.n_gram_size))
            
            if new_ngrams:
                session.add_all(new_ngrams)
                session.commit()
                for ngram in new_ngrams:
                    session.refresh(ngram)
                    ngram_cache[ngram.text] = ngram.id
            
            links = []
            for text_val, count in ngram_counts.items():
                links.append(DocumentNGram(document_id=db_doc.id, ngram_id=ngram_cache[text_val], count=count))
            session.add_all(links)
            session.commit()
            
            logger.info(f"Successfully processed document {db_doc.id}")

    logger.info("NLP processing complete.")

if __name__ == "__main__":
    main()
