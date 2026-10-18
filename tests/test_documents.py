from conftest import DOCS
from documents import chunk_all, chunk_doc, load_docs


def test_load_docs_reads_titles():
    docs = load_docs(DOCS)
    assert len(docs) == 12
    titles = {d["source"]: d["title"] for d in docs}
    assert titles["vpn-troubleshooting.md"] == "VPN: Connect and Troubleshoot"


def test_chunks_follow_sections_and_carry_title():
    doc = {"source": "x.md", "title": "Thing", "text": "# Thing\nIntro line.\n## Setup\nStep one.\nStep two.\n## Fix\nDo it."}
    chunks = chunk_doc(doc)
    assert [c["section"] for c in chunks] == ["Overview", "Setup", "Fix"]
    assert chunks[1]["text"] == "Thing - Setup\nStep one.\nStep two."  # line breaks kept
    assert all(c["source"] == "x.md" for c in chunks)


def test_long_section_split_with_overlap():
    body = " ".join(f"w{i}" for i in range(400))
    chunks = chunk_doc({"source": "x.md", "title": "T", "text": f"# T\n## Long\n{body}"}, max_words=100, overlap_words=20)
    assert len(chunks) == 5
    first, second = chunks[0]["text"].split()[3:], chunks[1]["text"].split()[3:]
    assert first[-20:] == second[:20]  # windows overlap
    assert chunks[-1]["text"].split()[-1] == "w399"  # nothing lost at the end


def test_all_docs_chunk_ids_unique():
    chunks = chunk_all(load_docs(DOCS))
    assert len({c["id"] for c in chunks}) == len(chunks) > 30
