# nlp_news
old style NLP mixed with databases

uv run scripts/import_complex.py --start-sample-id 0 --end-sample-id 16
uv run scripts/process_documents.py 
uv run scripts/cluster_documents.py 
uv run scripts/visualize_clusters.py
uv run scripts/serve.py
