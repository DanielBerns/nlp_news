# scripts/cluster_documents.py
import argparse
import pandas as pd
import logging
import sys
import os
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.ensemble import IsolationForest
from mlxtend.preprocessing import TransactionEncoder
from sqlmodel import Session, select
from nlp_news.config import load_config, get_engine, setup_logging
from nlp_news.models import Itemset, NGram, Document

logger = logging.getLogger("cluster_docs")

def generate_dendrogram(df, output_path="dendrogram.png"):
    """Calculates distances and saves a dendrogram image."""
    logger.info("Calculating linkage matrix for dendrogram...")
    # Calculate pairwise cosine distances
    dist_matrix = pdist(df, metric='cosine')
    # Generate the linkage matrix using average linkage (compatible with cosine)
    Z = hierarchy.linkage(dist_matrix, method='average')

    logger.info("Plotting dendrogram...")
    plt.figure(figsize=(14, 8))
    # Truncating the tree slightly so it doesn't become a massive black blob for thousands of docs
    hierarchy.dendrogram(Z, truncate_mode='level', p=10, leaf_rotation=90, leaf_font_size=8)
    plt.title("Hierarchical Clustering Dendrogram (Cosine Distance)")
    plt.xlabel("Document Index (or Cluster Size if truncated)")
    plt.ylabel("Distance Threshold")

    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    logger.info(f"Dendrogram saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Cluster documents based on itemsets.")
    parser.add_argument(
        "--method",
        choices=["dbscan", "hierarchical", "isolation_forest"],
        default="dbscan",
        help="Clustering algorithm to use (default: dbscan)"
    )
    args = parser.parse_args()

    setup_logging()
    config = load_config()
    engine = get_engine(config.db_path)

    logger.info(f"Starting clustering using strategy: {args.method.upper()}")

    with Session(engine) as session:
        # Load itemsets
        db_itemsets = session.exec(select(Itemset)).all()
        if not db_itemsets:
            logger.warning("No itemsets found. Run process_itemsets.py first.")
            return

        ngrams = session.exec(select(NGram)).all()
        ngram_map = {n.id: n.text for n in ngrams}

        doc_ids = []
        transactions = []
        for itemset in db_itemsets:
            doc_ids.append(itemset.document_id)
            transactions.append([ngram_map[i] for i in itemset.items])

        logger.info("Encoding transactions...")
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        df = pd.DataFrame(te_ary, columns=te.columns_)
        logger.info(f"Clustering matrix shape: {df.shape}")

        # Apply the chosen clustering strategy
        if args.method == "dbscan":
            eps = getattr(config, 'dbscan_eps', 0.3)
            min_samples = getattr(config, 'dbscan_min_samples', 3)
            model = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
            cluster_labels = model.fit_predict(df)

        elif args.method == "hierarchical":
            threshold = getattr(config, 'hierarchical_threshold', 0.6)
            model = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=threshold,
                metric='cosine',
                linkage='average'
            )
            cluster_labels = model.fit_predict(df)
            # Generate the visualization
            generate_dendrogram(df, "dendrogram.png")

        elif args.method == "isolation_forest":
            contamination = getattr(config, 'isolation_contamination', 0.05)
            model = IsolationForest(contamination=contamination, random_state=42)
            cluster_labels = model.fit_predict(df)

        # Statistics
        unique_clusters = set(cluster_labels)
        num_clusters = len(unique_clusters) - (1 if -1 in unique_clusters else 0)
        num_noise = list(cluster_labels).count(-1)

        logger.info(f"Clustering complete. Found {num_clusters} clusters (or normal classes).")
        logger.info(f"Classified {num_noise} documents as noise/anomalies (-1).")

        logger.info("Saving cluster assignments to database...")
        for doc_id, label in zip(doc_ids, cluster_labels):
            doc = session.get(Document, doc_id)
            if doc:
                doc.cluster_id = int(label)
                session.add(doc)

        session.commit()
        logger.info("Database updated successfully.")

if __name__ == "__main__":
    main()
