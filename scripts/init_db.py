import logging
from nlp_news.config import load_config, init_db, setup_logging

logger = logging.getLogger("init_db")

def main():
    setup_logging()
    try:
        config = load_config()
        logger.info(f"Initializing database at: {config.db_path}")
        init_db(config.db_path)
        logger.info("Database initialization complete.")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")

if __name__ == "__main__":
    main()
