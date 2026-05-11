import spacy
import logging
from sqlmodel import Session, select, text
from nlp_news.config import load_config, get_engine, load_spacy_model, setup_logging
from nlp_news.models import Document, NGram, DocumentNGram, Itemset

logger = logging.getLogger("process_itemsets")

def is_boring(text_val: str, nlp):
    doc = nlp(text_val.lower())
    boring = all(token.is_stop or token.is_punct for token in doc)
    if boring:
        logger.debug(f"N-gram '{text_val}' is classified as boring")
    return boring

def main():
    setup_logging()
    config = load_config()
    logger.info(f"Starting itemset generation. Language: {config.language}")
    
    engine = get_engine(config.db_path)
    nlp = load_spacy_model(config.language)
    
    with Session(engine) as session:
        logger.info("Clearing existing itemsets")
        session.execute(text("DELETE FROM itemset"))
        session.commit()
        
        documents = session.exec(select(Document)).all()
        logger.info(f"Processing {len(documents)} documents for itemsets")
        
        for doc in documents:
            logger.info(f"Generating itemset for {doc.path}")
            links = session.exec(
                select(DocumentNGram, NGram)
                .join(NGram)
                .where(DocumentNGram.document_id == doc.id)
            ).all()
            
            interesting_ngram_ids = []
            for link, ngram in links:
                if not is_boring(ngram.text, nlp):
                    interesting_ngram_ids.append(ngram.id)
            
            logger.info(f"Found {len(interesting_ngram_ids)} interesting n-grams out of {len(links)} total")
            
            if interesting_ngram_ids:
                itemset = Itemset(document_id=doc.id, items=interesting_ngram_ids)
                session.add(itemset)
                logger.debug(f"Saved itemset for document ID {doc.id}")
        
        session.commit()
        logger.info("Itemset generation complete")

if __name__ == "__main__":
    main()
