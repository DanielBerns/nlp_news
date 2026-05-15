import os
import json
import logging
from collections import defaultdict
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from sqlmodel import Session, select, func
import uvicorn

from nlp_news.config import load_config, get_engine, setup_logging
from nlp_news.models import AssociationRule, Document, NGram, DocumentNGram

logger = logging.getLogger("serve")

@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    logger.info("Starting web server...")
    yield

app = FastAPI(lifespan=lifespan)
config = load_config()
engine = get_engine(config.db_path)

def get_base_html(title: str, body_content: str) -> str:
    """Provides a consistent layout and navigation bar for all pages."""
    return f"""
    <html>
        <head>
            <title>{title}</title>
            <style>
                body {{ font-family: sans-serif; margin: 20px; line-height: 1.6; color: #333; }}
                nav {{ margin-bottom: 20px; padding: 15px; background: #f4f4f4; border-radius: 5px; }}
                nav a {{ margin-right: 20px; text-decoration: none; color: #0066cc; font-weight: bold; }}
                nav a:hover {{ text-decoration: underline; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #e9e9e9; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .doc-card {{ background: #fdfdfd; border: 1px solid #ddd; padding: 15px; margin-bottom: 15px; border-radius: 5px; }}
                pre {{ white-space: pre-wrap; font-family: inherit; margin: 0; }}
                .cluster-section {{ margin-top: 30px; border-top: 2px solid #ccc; padding-top: 10px; }}
                ul li {{ margin-bottom: 8px; }}
                .btn-back {{ display: inline-block; margin-top: 20px; padding: 8px 15px; background: #eee; text-decoration: none; border-radius: 3px; color: #333; }}
                .img-container {{ max-width: 100%; overflow-x: auto; border: 1px solid #ccc; padding: 10px; background: white; }}
            </style>
        </head>
        <body>
            <nav>
                <a href="/">Dashboard (Rules)</a>
                <a href="/documents">All Documents</a>
                <a href="/clusters">Browse by Clusters</a>
                <a href="/bigrams">Browse by Bigrams</a>
                <a href="/dendrogram">View Dendrogram</a>
            </nav>
            {body_content}
        </body>
    </html>
    """

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Home page showing the mined association rules."""
    logger.info("Received request for index")
    with Session(engine) as session:
        rules = session.exec(select(AssociationRule).order_by(AssociationRule.lift.desc())).all()

    rules_html = "".join([
        f"<tr><td>{', '.join(r.antecedents)}</td><td>=></td><td>{', '.join(r.consequents)}</td>"
        f"<td>{r.support:.2f}</td><td>{r.confidence:.2f}</td><td>{r.lift:.2f}</td></tr>"
        for r in rules
    ])

    content = f"""
    <h1>Association Rules</h1>
    <table>
        <tr><th>Antecedents</th><th></th><th>Consequents</th><th>Support</th><th>Confidence</th><th>Lift</th></tr>
        {rules_html}
    </table>
    """
    return get_base_html("NLP News - Dashboard", content)

@app.get("/documents", response_class=HTMLResponse)
async def view_documents():
    """Lists all documents."""
    with Session(engine) as session:
        docs = session.exec(select(Document).order_by(Document.timestamp.desc())).all()

    docs_html = "".join([
        f"<li><strong>{d.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</strong> - <a href='/doc/{d.id}'>{d.path}</a></li>"
        for d in docs
    ])
    content = f"<h1>All Documents</h1><ul>{docs_html}</ul>"
    return get_base_html("All Documents", content)

@app.get("/clusters", response_class=HTMLResponse)
async def view_clusters():
    """Groups documents by their cluster_id."""
    with Session(engine) as session:
        docs = session.exec(select(Document)).all()

    clusters = defaultdict(list)
    for d in docs:
        # Group unclustered documents into -1
        cid = d.cluster_id if d.cluster_id is not None else -1
        clusters[cid].append(d)

    content = "<h1>Document Clusters</h1>"
    for cid in sorted(clusters.keys()):
        c_name = "Noise (-1)" if cid == -1 else f"Cluster {cid}"
        content += f"<div class='cluster-section'><h2>{c_name} ({len(clusters[cid])} docs)</h2><ul>"
        for d in clusters[cid]:
            content += f"<li><a href='/doc/{d.id}'>{d.path}</a></li>"
        content += "</ul></div>"

    return get_base_html("Clusters", content)

@app.get("/bigrams", response_class=HTMLResponse)
async def view_bigrams():
    """Lists the top 100 most frequent bigrams."""
    with Session(engine) as session:
        # Join NGram and DocumentNGram to count document occurrences where n=2
        statement = select(NGram.id, NGram.text, func.count(DocumentNGram.document_id).label("doc_count"))\
            .join(DocumentNGram)\
            .where(NGram.n == 2)\
            .group_by(NGram.id, NGram.text)\
            .order_by(func.count(DocumentNGram.document_id).desc())\
            .limit(100)

        results = session.exec(statement).all()

    content = "<h1>Top 100 Bigrams</h1><p>Click a bigram to see which documents contain it.</p><ul>"
    for ngram_id, text, count in results:
        content += f"<li><a href='/bigram/{ngram_id}'><strong>'{text}'</strong></a> (Found in {count} documents)</li>"
    content += "</ul>"
    return get_base_html("Browse by Bigrams", content)

@app.get("/bigram/{ngram_id}", response_class=HTMLResponse)
async def view_bigram_docs(ngram_id: int):
    """Shows all documents that contain a specific bigram."""
    with Session(engine) as session:
        ngram = session.get(NGram, ngram_id)
        if not ngram:
            return HTMLResponse("Bigram not found", status_code=404)

        statement = select(Document).join(DocumentNGram).where(DocumentNGram.ngram_id == ngram_id)
        docs = session.exec(statement).all()

    content = f"<h1>Documents containing bigram: '{ngram.text}'</h1><ul>"
    for d in docs:
        content += f"<li><a href='/doc/{d.id}'>{d.path}</a></li>"
    content += "</ul>"
    content += "<br><a href='/bigrams' class='btn-back'>&laquo; Back to Bigrams</a>"
    return get_base_html(f"Bigram: {ngram.text}", content)

@app.get("/doc/{doc_id}", response_class=HTMLResponse)
async def view_document(doc_id: int):
    """Displays the full content and metadata of a single document."""
    with Session(engine) as session:
        doc = session.get(Document, doc_id)
        if not doc:
            return HTMLResponse("Document not found", status_code=404)

    cluster_text = f"Cluster {doc.cluster_id}" if doc.cluster_id is not None and doc.cluster_id != -1 else "Noise (-1)"
    if doc.cluster_id is None:
        cluster_text = "Unclustered (Run cluster_documents.py)"

    content = f"""
    <h1>{doc.path}</h1>
    <p><strong>Cluster assignment:</strong> {cluster_text}</p>
    <p><strong>Processed on:</strong> {doc.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
    <p><strong>Source URL:</strong> <a href='{doc.source_url}'>{doc.source_url}</a></p>

    <h2>Document Content</h2>
    <div class='doc-card'>
        <pre>{doc.content}</pre>
    </div>

    <a href="javascript:history.back()" class='btn-back'>&laquo; Go Back</a>
    """
    return get_base_html(f"Reading: {doc.path}", content)

@app.get("/api/dendrogram_data")
async def get_dendrogram_data():
    """Serves the JSON tree data to the frontend."""
    if os.path.exists("dendrogram.json"):
        with open("dendrogram.json", "r") as f:
            return JSONResponse(json.load(f))
    return JSONResponse({"error": "No data found"}, status_code=404)

@app.get("/dendrogram", response_class=HTMLResponse)
async def view_dendrogram():
    """Displays the interactive ECharts dendrogram."""
    if not os.path.exists("dendrogram.json"):
        content = """
        <h1>Interactive Dendrogram</h1>
        <p>No dendrogram data found. Run the clustering script with the hierarchical method first.</p>
        <p>Run: <code>python -m scripts.cluster_documents --method hierarchical</code></p>
        """
        return get_base_html("Dendrogram", content)

    content = """
    <h1>Interactive Document Taxonomy</h1>
    <p>Scroll to zoom, click and drag to pan. <strong>Click on a leaf node (Doc #) to read the document.</strong></p>

    <div id="tree-container" style="width: 100%; height: 800px; border: 1px solid #ccc; background: white; border-radius: 5px;"></div>

    <script src="https://cdn.jsdelivr.net/npm/echarts@5.5.0/dist/echarts.min.js"></script>

    <script>
        document.addEventListener("DOMContentLoaded", function() {
            var chartDom = document.getElementById('tree-container');
            var myChart = echarts.init(chartDom);

            myChart.showLoading();

            // Fetch the JSON tree data from our FastAPI endpoint
            fetch('/api/dendrogram_data')
                .then(response => response.json())
                .then(data => {
                    myChart.hideLoading();

                    var option = {
                        tooltip: { trigger: 'item', triggerOn: 'mousemove' },
                        series: [
                            {
                                type: 'tree',
                                data: [data], // Pass the nested JSON
                                top: '5%', left: '10%', bottom: '5%', right: '20%',
                                symbolSize: 10,
                                roam: true, // Enables zooming and panning
                                label: {
                                    position: 'left',
                                    verticalAlign: 'middle',
                                    align: 'right',
                                    fontSize: 12
                                },
                                leaves: {
                                    label: {
                                        position: 'right',
                                        verticalAlign: 'middle',
                                        align: 'left',
                                        // Highlight clickable documents
                                        color: '#0066cc',
                                        fontWeight: 'bold'
                                    }
                                },
                                emphasis: { focus: 'descendant' },
                                expandAndCollapse: true,
                                animationDuration: 550,
                                animationDurationUpdate: 750
                            }
                        ]
                    };
                    myChart.setOption(option);

                    // Add the Click Event Listener
                    myChart.on('click', function(params) {
                        // Check if the clicked node has a doc_id (meaning it's a leaf/document)
                        if (params.data && params.data.doc_id !== undefined) {
                            // Redirect the user to the document reading page
                            window.location.href = '/doc/' + params.data.doc_id;
                        }
                    });
                });
        });
    </script>
    """
    return get_base_html("Interactive Dendrogram", content)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
