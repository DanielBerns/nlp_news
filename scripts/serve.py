from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from sqlmodel import Session, select
from nlp_news.config import load_config, get_engine, setup_logging
from nlp_news.models import AssociationRule, Document
import uvicorn
import logging

logger = logging.getLogger("serve")

app = FastAPI()
config = load_config()
engine = get_engine(config.db_path)

@app.on_event("startup")
def on_startup():
    setup_logging()
    logger.info("Starting web server...")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    logger.info("Received request for index")
    with Session(engine) as session:
        rules = session.exec(select(AssociationRule).order_by(AssociationRule.lift.desc())).all()
        docs = session.exec(select(Document).order_by(Document.timestamp.desc())).all()
        
    logger.info(f"Retrieved {len(rules)} rules and {len(docs)} documents")
    
    rules_html = "".join([
        f"<tr><td>{', '.join(r.antecedents)}</td><td>=></td><td>{', '.join(r.consequents)}</td>"
        f"<td>{r.support:.2f}</td><td>{r.confidence:.2f}</td><td>{r.lift:.2f}</td></tr>"
        for r in rules
    ])
    
    docs_html = "".join([
        f"<li><strong>{d.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</strong> - <a href='{d.source_url}'>{d.path}</a><br/><pre>{d.content[:200]}...</pre></li>"
        for d in docs
    ])
    
    content = f"""
    <html>
        <head>
            <title>NLP News - Association Rules</title>
            <style>
                body {{ font-family: sans-serif; margin: 20px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2 f2 f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                pre {{ background: #f4f4f4; padding: 10px; white-space: pre-wrap; }}
            </style>
        </head>
        <body>
            <h1>Association Rules</h1>
            <table>
                <tr><th>Antecedents</th><th></th><th>Consequents</th><th>Support</th><th>Confidence</th><th>Lift</th></tr>
                {rules_html}
            </table>
            <h1>Documents</h1>
            <ul>{docs_html}</ul>
        </body>
    </html>
    """
    return content

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
