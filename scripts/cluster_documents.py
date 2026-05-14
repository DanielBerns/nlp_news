# scripts/cluster_documents.py
import pandas as pd
import logging
import sys
from sklearn.cluster import DBSCAN
from mlxtend.preprocessing import TransactionEncoder
from sqlmodel import Session, select
from nlp_news.config import load_config, get_engine, setup_logging
from nlp_news.models import Itemset, NGram, Document

logger = logging.getLogger("cluster_docs")

def main():
    setup_logging()
    config = load_config()

    # Safely get DBSCAN parameters, providing defaults if they aren't in config yet
    eps = getattr(config, 'dbscan_eps', 0.3)
    min_samples = getattr(config, 'dbscan_min_samples', 3)

    logger.info(f"Starting DBSCAN clustering. eps: {eps}, min_samples: {min_samples}")

    engine = get_engine(config.db_path)

    with Session(engine) as session:
        logger.info("Loading itemsets from database")
        db_itemsets = session.exec(select(Itemset)).all()
        if not db_itemsets:
            logger.warning("No itemsets found. Run process_itemsets.py first.")
            return

        ngrams = session.exec(select(NGram)).all()
        ngram_map = {n.id: n.text for n in ngrams}

        # Track doc_ids so we can map the resulting labels back to the database
        doc_ids = []
        transactions = []
        for itemset in db_itemsets:
            doc_ids.append(itemset.document_id)
            transactions.append([ngram_map[i] for i in itemset.items])

        logger.info("Encoding transactions for mining")
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        df = pd.DataFrame(te_ary, columns=te.columns_)
        logger.info(f"Clustering matrix shape: {df.shape}")

        # Initialize DBSCAN.
        # Using metric='cosine' is crucial here. Since our matrix is composed of 0s and 1s,
        # cosine distance evaluates the angle between the vectors, which is much more
        # effective for text itemsets than Euclidean distance.
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
        cluster_labels = dbscan.fit_predict(df)

        # Calculate some statistics to log
        # DBSCAN labels outliers/noise as -1
        num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        num_noise = list(cluster_labels).count(-1)

        logger.info(f"Clustering complete. Found {num_clusters} clusters.")
        logger.info(f"Classified {num_noise} documents as noise (outliers).")

        logger.info("Saving cluster assignments back to the Document table...")
        for doc_id, label in zip(doc_ids, cluster_labels):
            doc = session.get(Document, doc_id)
            if doc:
                # Store the label. -1 means the document didn't fit into any cluster.
                doc.cluster_id = int(label)
                session.add(doc)

        session.commit()
        logger.info("Database updated successfully.")

if __name__ == "__main__":
    main()
