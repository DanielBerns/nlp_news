import os
import logging
import json
import argparse
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from bs4 import BeautifulSoup
from sqlmodel import Session, select
from nlp_news.config import load_config, get_engine, setup_logging
from nlp_news.models import Document

logger = logging.getLogger("import_complex")

def parse_attr_timestamp(ts_str: str):
    try:
        # Format: 20240402210052 -> YYYYMMDDHHMMSS
        return datetime.strptime(ts_str, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    except Exception as e:
        logger.debug(f"Failed to parse attribute timestamp {ts_str}: {e}")
        return None

def load_attributes(directory: str):
    attr_path = os.path.join(directory, "attributes.json")
    if os.path.exists(attr_path):
        try:
            with open(attr_path, "r") as f:
                data = json.load(f)
            return {
                "timestamp": parse_attr_timestamp(data.get("timestamp", "")),
                "url": data.get("response.url") or data.get("original url")
            }
        except Exception as e:
            logger.warning(f"Failed to load attributes from {attr_path}: {e}")
    return None

def extract_from_xml(content: str):
    soup = BeautifulSoup(content, "lxml-xml")
    items = soup.find_all("item")
    results = []
    for item in items:
        text = None
        desc = item.find("description")
        if desc and desc.text:
            text = BeautifulSoup(desc.text, "html.parser").get_text(separator=" ")
        else:
            title = item.find("title")
            if title:
                text = title.text
        
        if text and len(text.strip()) > 20:
            link = item.find("link")
            pub_date = item.find("pubDate")
            
            ts = None
            if pub_date:
                try:
                    ts = parsedate_to_datetime(pub_date.text)
                except:
                    pass
            
            results.append({
                "text": text.strip(),
                "url": link.text.strip() if link else None,
                "timestamp": ts
            })
    return results

def extract_from_html(content: str):
    soup = BeautifulSoup(content, "html.parser")
    
    canonical = soup.find("link", rel="canonical")
    url = canonical["href"] if canonical and canonical.has_attr("href") else None
    
    pub_time = soup.find("meta", property="article:published_time")
    ts = None
    if pub_time and pub_time.has_attr("content"):
        try:
            ts = datetime.fromisoformat(pub_time["content"])
        except:
            pass

    for script_or_style in soup(["script", "style", "nav", "footer", "header"]):
        script_or_style.decompose()
    
    main = soup.find("main") or soup.find("article") or soup.find("div", class_="content")
    if main:
        text = main.get_text(separator=" ")
    else:
        text = soup.get_text(separator=" ")
        
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = "\n".join(chunk for chunk in chunks if chunk)
    
    if len(text) > 100:
        return [{"text": text, "url": url, "timestamp": ts}]
    return []

def main():
    parser = argparse.ArgumentParser(description="Import data from specific sample directories")
    parser.add_argument("--start-sample-id", type=int, default=0, help="Starting sample ID (inclusive)")
    parser.add_argument("--end-sample-id", type=int, default=10, help="Ending sample ID (exclusive)")
    args = parser.parse_args()

    setup_logging()
    config = load_config()
    engine = get_engine(config.db_path)
    
    base_path = "data/samples"
    logger.info(f"Scanning samples from {args.start_sample_id} to {args.end_sample_id}...")
    
    imported_count = 0
    processed_files_count = 0
    
    with Session(engine) as session:
        for sample_id in range(args.start_sample_id, args.end_sample_id):
            level_1 = sample_id // 65536
            level_2 = (sample_id // 256) % 256
            level_3 = sample_id % 256
            sample_dir = os.path.join(base_path, f"{level_1:03d}", f"{level_2:03d}", f"{level_3:03d}")
            
            if not os.path.exists(sample_dir):
                continue
                
            for root, dirs, files in os.walk(sample_dir):
                # Load attributes for the current directory
                attrs = load_attributes(root)
                
                for filename in files:
                    if not (filename.endswith(".xml") or filename.endswith(".html") or filename.endswith(".htm")):
                        continue
                    
                    processed_files_count += 1
                    file_path = os.path.join(root, filename)
                    try:
                        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                            content = f.read()
                        
                        extracted_data = []
                        if filename.endswith(".xml"):
                            extracted_data = extract_from_xml(content)
                        else:
                            extracted_data = extract_from_html(content)
                        
                        for item in extracted_data:
                            text = item["text"]
                            snippet_id = abs(hash(text)) % 1000000
                            pseudo_path = f"{file_path}#{snippet_id}"
                            
                            existing = session.exec(select(Document).where(Document.path == pseudo_path)).first()
                            if not existing:
                                # Prefer attributes.json over embedded metadata
                                final_url = (attrs["url"] if attrs and attrs["url"] else item["url"])
                                final_ts = (attrs["timestamp"] if attrs and attrs["timestamp"] else item["timestamp"])
                                
                                if not final_ts:
                                    mtime = os.path.getmtime(file_path)
                                    final_ts = datetime.fromtimestamp(mtime, tz=timezone.utc)
                                
                                doc = Document(
                                    path=pseudo_path, 
                                    content=text, 
                                    source_url=final_url, 
                                    timestamp=final_ts
                                )
                                session.add(doc)
                                imported_count += 1
                        
                        if len(extracted_data) > 0:
                            logger.info(f"Imported {len(extracted_data)} snippets from {file_path}")
                            session.commit()
                            
                    except Exception as e:
                        logger.error(f"Failed to process {file_path}: {e}")

    logger.info(f"Import complete. Total new documents added: {imported_count}")

if __name__ == "__main__":
    main()
