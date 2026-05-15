# scripts/visualize_clusters.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from sklearn.decomposition import PCA
from mlxtend.preprocessing import TransactionEncoder
from sqlmodel import Session, select
from nlp_news.config import load_config, get_engine, setup_logging
from nlp_news.models import Itemset, NGram, Document

logger = logging.getLogger("visualize_clusters")

def main():
    setup_logging()
    config = load_config()
    logger.info("Starting cluster visualization...")

    engine = get_engine(config.db_path)

    with Session(engine) as session:
        # 1. Fetch clustered documents
        documents = session.exec(
            select(Document).where(Document.cluster_id != None)
        ).all()

        if not documents:
            logger.warning("No clustered documents found. Run cluster_documents.py first.")
            return

        # Map document IDs to their cluster labels
        doc_cluster_map = {doc.id: doc.cluster_id for doc in documents}

        # 2. Fetch n-grams and itemsets
        db_itemsets = session.exec(select(Itemset)).all()
        ngrams = session.exec(select(NGram)).all()
        ngram_map = {n.id: n.text for n in ngrams}

        transactions = []
        labels = []

        # Reconstruct the text transactions and align them with the correct cluster labels
        for itemset in db_itemsets:
            if itemset.document_id in doc_cluster_map:
                transactions.append([ngram_map[i] for i in itemset.items])
                labels.append(doc_cluster_map[itemset.document_id])

        if not transactions:
            logger.warning("No itemsets matching the clustered documents were found.")
            return

        # 3. Build the binary feature matrix
        logger.info("Encoding transactions for dimensionality reduction...")
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        df = pd.DataFrame(te_ary, columns=te.columns_)

        # 4. Dimensionality Reduction (PCA)
        logger.info("Running PCA to reduce dimensions to 2D...")
        pca = PCA(n_components=2, random_state=42)
        components = pca.fit_transform(df)

        # 5. Prepare Data for Plotting
        plot_df = pd.DataFrame({
            'PCA1': components[:, 0],
            'PCA2': components[:, 1],
            'Cluster': labels
        })

        # Format the cluster names for the legend (marking -1 explicitly as noise)
        plot_df['Cluster Name'] = plot_df['Cluster'].apply(
            lambda x: 'Noise (-1)' if x == -1 else f'Cluster {x}'
        )

        # 6. Generate the Scatter Plot
        logger.info("Generating scatter plot...")
        plt.figure(figsize=(12, 8))

        # Create a custom color palette (keep Noise as grey, use vibrant colors for others)
        unique_clusters = sorted(plot_df['Cluster Name'].unique())
        palette = sns.color_palette("husl", len(unique_clusters))
        color_dict = {cluster: color for cluster, color in zip(unique_clusters, palette)}

        if 'Noise (-1)' in color_dict:
            color_dict['Noise (-1)'] = (0.7, 0.7, 0.7)  # Override noise with grey

        sns.scatterplot(
            data=plot_df,
            x='PCA1',
            y='PCA2',
            hue='Cluster Name',
            palette=color_dict,
            alpha=0.7,
            edgecolor=None
        )

        # Styling
        plt.title('Document Clusters (2D PCA Projection of Itemsets)', fontsize=14)
        plt.xlabel('Principal Component 1', fontsize=12)
        plt.ylabel('Principal Component 2', fontsize=12)

        # Move legend outside the plot
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Clusters')
        # plt.tight_layout()

        # 7. Save the Output
        output_file = 'clusters_visualization.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"Visualization saved successfully to {output_file}")

if __name__ == "__main__":
    main()
