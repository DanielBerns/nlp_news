import pandas as pd
import logging
import sys
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from sqlmodel import Session, select, text
from nlp_news.config import load_config, get_engine, setup_logging
from nlp_news.models import Itemset, NGram, AssociationRule

logger = logging.getLogger("mine_rules")

def main():
    setup_logging()
    try:
        config = load_config()
        logger.info(f"Starting rule mining. Min Support: {config.min_support}, Min Confidence: {config.min_confidence}")
        
        engine = get_engine(config.db_path)
        
        with Session(engine) as session:
            logger.info("Clearing existing association rules")
            session.execute(text("DELETE FROM associationrule"))
            session.commit()

            logger.info("Loading itemsets from database")
            db_itemsets = session.exec(select(Itemset)).all()
            if not db_itemsets:
                logger.warning("No itemsets found. Skipping rule mining.")
                return
            
            logger.info(f"Loading {len(db_itemsets)} transactions")
            ngrams = session.exec(select(NGram)).all()
            ngram_map = {n.id: n.text for n in ngrams}
            
            transactions = []
            for itemset in db_itemsets:
                transactions.append([ngram_map[i] for i in itemset.items])
            
            logger.info("Encoding transactions for mining")
            te = TransactionEncoder()
            te_ary = te.fit(transactions).transform(transactions)
            df = pd.DataFrame(te_ary, columns=te.columns_)
            logger.info(f"Mining matrix shape: {df.shape}")
            
            logger.info(f"Executing Apriori with min_support={config.min_support}")
            # Note: Using apriori for now as per previous attempt, fpgrowth was also 137
            frequent_itemsets = apriori(df, min_support=config.min_support, use_colnames=True)
            
            if frequent_itemsets.empty:
                logger.info("No frequent itemsets discovered.")
                return
            
            logger.info(f"Found {len(frequent_itemsets)} frequent itemsets")
                
            logger.info(f"Generating association rules with min_confidence={config.min_confidence}")
            rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=config.min_confidence)
            
            logger.info(f"Mined {len(rules)} rules. Storing in database...")
            for _, row in rules.iterrows():
                rule = AssociationRule(
                    antecedents=list(row['antecedents']),
                    consequents=list(row['consequents']),
                    support=row['support'],
                    confidence=row['confidence'],
                    lift=row['lift']
                )
                session.add(rule)
            
            session.commit()
            logger.info("Rule mining and storage complete")
            
    except Exception as e:
        logger.exception("A critical error occurred during rule mining")
        sys.exit(1)

if __name__ == "__main__":
    main()
