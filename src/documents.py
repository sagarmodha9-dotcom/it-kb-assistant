"""Load the knowledge base documents and split them into chunks for search.

Each document is a Markdown file that starts with "# Title" and uses "## " section headings.
Chunks follow the sections, so a chunk is one complete topic ("Account locked") instead of an
arbitrary slice of text. Long sections are split into overlapping windows. Every chunk starts
with the document title and section name, so the embedding knows what the chunk is about.

(The deployed Bedrock Knowledge Base does its own chunking during ingestion; this module is used
for the local vector index and the keyword baseline.)
"""
import os
import re


def load_docs(directory: str) -> list[dict]:
    docs = []
    for name in sorted(os.listdir(directory)):
        if not name.endswith(".md"):
            continue
        with open(os.path.join(directory, name), encoding="utf-8") as f:
            text = f.read()
        match = re.search(r"^# (.+)$", text, re.M)
        docs.append({"source": name, "title": match.group(1).strip() if match else name, "text": text})
    return docs


def chunk_doc(doc: dict, max_words: int = 160, overlap_words: int = 30) -> list[dict]:
    sections = re.split(r"^## ", doc["text"], flags=re.M)
    intro = re.sub(r"^# .+$", "", sections[0], flags=re.M).strip()
    parts = [("Overview", intro)] if intro else []
    for section in sections[1:]:
        heading, _, body = section.partition("\n")
        parts.append((heading.strip(), body.strip()))

    chunks = []
    for heading, body in parts:
        words = body.split()
        if not words:
            continue
        if len(words) <= max_words:  # keep short sections as written (line breaks help the model)
            chunks.append({"id": f"{doc['source']}#{len(chunks)}", "source": doc["source"],
                           "title": doc["title"], "section": heading,
                           "text": f"{doc['title']} - {heading}\n{body}"})
            continue
        step = max_words - overlap_words
        for start in range(0, max(len(words) - overlap_words, 1), step):
            piece = " ".join(words[start:start + max_words])
            chunks.append({
                "id": f"{doc['source']}#{len(chunks)}",
                "source": doc["source"],
                "title": doc["title"],
                "section": heading,
                "text": f"{doc['title']} - {heading}\n{piece}",
            })
    return chunks


def chunk_all(docs: list[dict], **kwargs) -> list[dict]:
    return [chunk for doc in docs for chunk in chunk_doc(doc, **kwargs)]
