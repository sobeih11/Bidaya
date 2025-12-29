import os
import json
import logging
import base64
import time
import glob
import hashlib
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pymupdf as fitz  # PyMuPDF
from dotenv import load_dotenv
from openai import OpenAI


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logging.warning("OPENAI_API_KEY environment variable not set. LLM calls will fail.")

VISION_MODEL_NAME = os.getenv("OPENAI_VISION_MODEL", "gpt-4o-mini")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("generic_ocr_logs.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)


# -----------------------------------------------------------------------------
# Helpers: chunking + headers
# -----------------------------------------------------------------------------
HEADER_ALLCAPS_RE = re.compile(r"^[A-Z0-9\s\-\–\—\(\)\.:]{6,}$")
NOTE_RE = re.compile(r"^(NOTE|NOTES)\s+\d+(\.\d+)?", re.IGNORECASE)
NUM_HEADING_RE = re.compile(r"^\d+(\.\d+){0,3}\s+\S+")

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def normalize_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def is_header_like(text: str) -> bool:
    t = normalize_spaces(text)
    if not t:
        return False
    if NOTE_RE.match(t):
        return True
    if HEADER_ALLCAPS_RE.match(t):
        return True
    if NUM_HEADING_RE.match(t) and len(t.split()) <= 14:
        return True
    # short + title-ish
    if len(t.split()) <= 10 and not re.search(r"\d{4,}", t):
        # كثير من العناوين العربية بتطلع قصيرة بدون أرقام
        return True
    return False

def detect_header_level(text: str) -> int:
    """
    Heuristic levels:
    - NOTE X / NOTES X => level 1
    - 1. / 1.1 / ...   => based on dot depth (level 1..4)
    - ALL CAPS / short => level 2
    """
    t = normalize_spaces(text)
    if NOTE_RE.match(t):
        return 1
    m = re.match(r"^(\d+(?:\.\d+){0,3})\s+", t)
    if m:
        depth = m.group(1).count(".")
        return min(1 + depth, 4)
    if HEADER_ALLCAPS_RE.match(t):
        return 2
    return 2

def chunk_text_by_chars(text: str, max_chars: int = 1800, overlap_chars: int = 150) -> List[str]:
    """
    Simple char-based chunking to keep it deterministic for demos.
    """
    t = (text or "").strip()
    if not t:
        return []
    if len(t) <= max_chars:
        return [t]
    chunks = []
    start = 0
    while start < len(t):
        end = min(start + max_chars, len(t))
        chunk = t[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(t):
            break
        start = max(0, end - overlap_chars)
    return chunks

def table_to_text(title: Optional[str], columns: List[str], rows: List[List[str]]) -> str:
    """
    Flatten table for embedding/retrieval (while keeping structured rows in JSON too).
    """
    title = title or ""
    cols = " | ".join([c.strip() for c in columns if c is not None])
    lines = []
    if title:
        lines.append(f"TABLE TITLE: {title}")
    if cols:
        lines.append(f"COLUMNS: {cols}")
    for r in rows[:200]:  # safety cap
        lines.append("ROW: " + " | ".join([str(x).strip() for x in r]))
    return "\n".join(lines).strip()

def build_summary_for_text(text: str, max_len: int = 220) -> str:
    """
    Demo-friendly: take first sentence/line-ish.
    """
    t = normalize_spaces(text)
    if not t:
        return ""
    # split by common sentence ends
    parts = re.split(r"[\.!\؟\?]\s+", t)
    s = parts[0].strip() if parts else t
    return (s[:max_len] + "…") if len(s) > max_len else s

def build_summary_for_table(header_path: List[str], table_title: Optional[str]) -> str:
    hp = " > ".join([normalize_spaces(x) for x in header_path if x])
    tt = normalize_spaces(table_title or "")
    if tt and hp:
        return f"Table '{tt}' under section '{hp}'."
    if tt:
        return f"Table '{tt}'."
    if hp:
        return f"Table under section '{hp}'."
    return "Table extracted from the document."


# -----------------------------------------------------------------------------
# Main Parser
# -----------------------------------------------------------------------------
class GenericVisionOCRParser:
    """
    Scanned-PDF parser:
    - Convert each page to image (high DPI for demo)
    - Vision returns STRICT JSON per page (blocks: header/text/table/image)
    - Build header-based chunks and save JSONL ready for RAG ingestion
    """

    def __init__(
        self,
        output_dir: str = "ocr_output",
        batch_size: int = 1,   # demo default: 1 = أعلى دقة
        dpi: int = 300,        # demo default
        max_text_chunk_chars: int = 1800,
        overlap_chars: int = 150,
    ):
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.dpi = dpi
        self.max_text_chunk_chars = max_text_chunk_chars
        self.overlap_chars = overlap_chars

        self.checkpoints_dir = os.path.join(output_dir, "checkpoints")
        self.raw_json_dir = os.path.join(output_dir, "raw_pages_json")
        self.images_dir = os.path.join(output_dir, "images")
        self.chunks_dir = os.path.join(output_dir, "chunks_jsonl")
        self.manifests_dir = os.path.join(output_dir, "manifests")

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        os.makedirs(self.raw_json_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.chunks_dir, exist_ok=True)
        os.makedirs(self.manifests_dir, exist_ok=True)

        self.client = OpenAI(api_key=OPENAI_API_KEY)
        logging.info(f"Initialized with model: {VISION_MODEL_NAME}, dpi={dpi}, batch_size={batch_size}")

    @staticmethod
    def _png_bytes_to_data_url(img_bytes: bytes) -> str:
        b64 = base64.b64encode(img_bytes).decode("utf-8")
        return f"data:image/png;base64,{b64}"

    def _get_extraction_prompt(self, doc_title: str, page_start: int, page_end: int) -> str:
        """
        IMPORTANT: return strict JSON only.
        Page numbers are 1-based in output.
        """
        return f"""
You are an OCR + document structure extraction engine.

Document title: "{doc_title}"
You will receive images for pages {page_start} to {page_end} (inclusive).
The document is scanned. Extract content in natural reading order.

RETURN STRICT JSON ONLY (no markdown fences, no extra text).

Schema:
{{
  "pages": [
    {{
      "page": <int>,  // 1-based
      "blocks": [
        {{
          "type": "header" | "text" | "table" | "image" | "signature" | "footer",
          "text": <string|null>,

          // For headers only:
          "header_level": <int|null>,   // 1..4 if you can infer, else null

          // For tables only:
          "table": {{
            "title": <string|null>,
            "columns": [<string>, ...],
            "rows": [[<string>, ...], ...]
          }} | null,

          // For images (non-table):
          "image": {{
            "type": <string|null>,
            "caption": <string|null>,
            "text_in_image": <string|null>
          }} | null
        }}
      ]
    }}
  ]
}}

Rules:
- Preserve Arabic correctly (RTL) and keep mixed Arabic/English order.
- Do NOT hallucinate missing text. If unreadable, put null or "".
- If you see a table, fill the table object (columns+rows). Keep numbers exactly (commas, parentheses, minus).
- If the page has a clear main header/title, emit it as type="header".
- If unsure header_level, set null.

Now extract:
""".strip()

    def _call_openai_with_retry(self, prompt: str, batch_images: List[bytes], max_retries: int = 5) -> str:
        delay = 2
        for attempt in range(max_retries):
            try:
                content = [{"type": "input_text", "text": prompt}]
                for img_bytes in batch_images:
                    content.append({"type": "input_image", "image_url": self._png_bytes_to_data_url(img_bytes)})

                resp = self.client.responses.create(
                    model=VISION_MODEL_NAME,
                    input=[{"role": "user", "content": content}],
                )

                text = getattr(resp, "output_text", None)
                if text is None:
                    try:
                        text = ""
                        for item in resp.output:
                            if getattr(item, "type", "") == "message":
                                for part in item.content:
                                    if getattr(part, "type", "") in ("output_text", "text"):
                                        text += getattr(part, "text", "") + "\n"
                        text = text.strip()
                    except Exception:
                        text = ""
                return text or ""

            except Exception as e:
                logging.warning(f"API call failed attempt {attempt + 1}/{max_retries}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(delay)
                    delay *= 2
                else:
                    logging.error("Max retries reached.")
                    return ""
        return ""

    # -------------------------------------------------------------------------
    # Checkpoint helpers
    # -------------------------------------------------------------------------
    def _get_checkpoint_path(self, pdf_path: str) -> str:
        base_name = Path(pdf_path).stem
        return os.path.join(self.checkpoints_dir, f"{base_name}_checkpoint.json")

    def _load_checkpoint(self, pdf_path: str) -> Dict[str, Any]:
        p = self._get_checkpoint_path(pdf_path)
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logging.warning(f"Failed to load checkpoint: {e}")
        return {"completed_pages": 0, "total_pages": 0}

    def _save_checkpoint(self, pdf_path: str, data: Dict[str, Any]):
        p = self._get_checkpoint_path(pdf_path)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    # -------------------------------------------------------------------------
    # Core: build chunks with header stack
    # -------------------------------------------------------------------------
    def _build_chunks_from_pages_json(
        self,
        doc_id: str,
        file_name: str,
        file_sha: str,
        page_count: int,
        pages_json: Dict[str, Any],
        dpi: int,
        parser_version: str = "demo-v1"
    ) -> List[Dict[str, Any]]:
        chunks: List[Dict[str, Any]] = []
        header_stack: List[str] = []

        def current_header_path() -> List[str]:
            return [h for h in header_stack if h]

        for page_obj in pages_json.get("pages", []):
            page_no = int(page_obj.get("page", 0)) or 0
            blocks = page_obj.get("blocks", []) or []

            # We'll accumulate text under current header until we hit another header or table.
            buffer_text = ""
            buffer_surrounding = []

            def flush_text_buffer():
                nonlocal buffer_text, buffer_surrounding
                t = buffer_text.strip()
                if not t:
                    buffer_text = ""
                    buffer_surrounding = []
                    return

                hp = current_header_path()
                section = hp[0] if len(hp) >= 1 else ""
                subsection = hp[-1] if len(hp) >= 1 else ""
                summary = build_summary_for_text(t)

                for idx, piece in enumerate(chunk_text_by_chars(t, self.max_text_chunk_chars, self.overlap_chars), start=1):
                    chunk_id = f"{doc_id}_p{page_no:03d}_t{len(chunks)+1:06d}"
                    chunks.append({
                        "chunk_id": chunk_id,
                        "doc_id": doc_id,
                        "source": {
                            "file_name": file_name,
                            "file_sha256": file_sha,
                            "parser_version": parser_version,
                        },
                        "loc": {
                            "page": page_no,
                            "page_count": page_count,
                            "bbox": None,
                            "dpi": dpi,
                        },
                        "content": {
                            "type": "text",
                            "text": piece,
                            "table": None,
                        },
                        "context": {
                            "header_path": hp,
                            "section": section,
                            "subsection": subsection,
                            "summary": summary,
                            "surrounding_text": normalize_spaces(" ".join(buffer_surrounding))[:600] if buffer_surrounding else "",
                        },
                        "tags": {
                            "lang": "ar",  # لو عندك detection لاحقًا غيّرها
                            "doc_type": "scanned_document",
                        }
                    })

                buffer_text = ""
                buffer_surrounding = []

            for b in blocks:
                btype = (b.get("type") or "").strip().lower()
                btext = (b.get("text") or "").strip()

                # Sometimes model doesn't tag headers, so we backstop:
                if btype in ("text", "header") and btext and is_header_like(btext):
                    # treat as header
                    flush_text_buffer()
                    lvl = b.get("header_level")
                    lvl = int(lvl) if isinstance(lvl, (int, float, str)) and str(lvl).isdigit() else detect_header_level(btext)
                    # stack update
                    if lvl < 1:
                        lvl = 2
                    # level-1 means keep only 0 items then append; level-2 keep 1 item etc
                    header_stack[:] = header_stack[: max(0, lvl - 1)]
                    header_stack.append(normalize_spaces(btext))
                    continue

                if btype in ("footer", "signature"):
                    # skip from chunking, but you could store separately if you want
                    continue

                if btype == "table" and b.get("table"):
                    # flush any pending text first
                    flush_text_buffer()

                    tbl = b["table"] or {}
                    title = tbl.get("title")
                    cols = tbl.get("columns") or []
                    rows = tbl.get("rows") or []

                    hp = current_header_path()
                    section = hp[0] if len(hp) >= 1 else ""
                    subsection = hp[-1] if len(hp) >= 1 else ""
                    summary = build_summary_for_table(hp, title)

                    # optional: split huge tables into row groups for better retrieval
                    row_group_size = 35
                    if len(rows) <= row_group_size:
                        row_groups = [rows]
                    else:
                        row_groups = [rows[i:i+row_group_size] for i in range(0, len(rows), row_group_size)]

                    for gi, g in enumerate(row_groups, start=1):
                        chunk_id = f"{doc_id}_p{page_no:03d}_tbl{len(chunks)+1:06d}"
                        chunks.append({
                            "chunk_id": chunk_id,
                            "doc_id": doc_id,
                            "source": {
                                "file_name": file_name,
                                "file_sha256": file_sha,
                                "parser_version": parser_version,
                            },
                            "loc": {
                                "page": page_no,
                                "page_count": page_count,
                                "bbox": None,
                                "dpi": dpi,
                            },
                            "content": {
                                "type": "table",
                                "text": table_to_text(title, cols, g),
                                "table": {
                                    "title": title,
                                    "columns": cols,
                                    "rows": g
                                }
                            },
                            "context": {
                                "header_path": hp,
                                "section": section,
                                "subsection": subsection,
                                "summary": summary,
                                "surrounding_text": "",
                            },
                            "tags": {
                                "lang": "ar",
                                "doc_type": "scanned_document",
                                "table_part": gi,
                                "table_parts_total": len(row_groups),
                            }
                        })
                    continue

                # default: normal text block
                if btext:
                    buffer_text += (btext + "\n")
                    # keep a little for surrounding_text
                    if len(buffer_surrounding) < 6:
                        buffer_surrounding.append(btext)

            # end of page
            flush_text_buffer()

        return chunks

    # -------------------------------------------------------------------------
    # Process single PDF
    # -------------------------------------------------------------------------
    def _process_single_pdf(self, pdf_path: str, resume_from_checkpoint: bool) -> int:
        if not os.path.exists(pdf_path):
            logging.error(f"PDF file not found: {pdf_path}")
            return 0

        base_name = Path(pdf_path).stem
        doc_title = base_name.replace("_", " ").strip()
        file_sha = sha256_file(pdf_path)
        doc_id = f"{base_name}_{file_sha[:10]}"

        checkpoint = self._load_checkpoint(pdf_path) if resume_from_checkpoint else {"completed_pages": 0, "total_pages": 0}

        doc = fitz.open(pdf_path)
        num_pages = len(doc)
        checkpoint["total_pages"] = num_pages

        pdf_images_dir = os.path.join(self.images_dir, base_name)
        os.makedirs(pdf_images_dir, exist_ok=True)

        raw_pages_path = os.path.join(self.raw_json_dir, f"{base_name}_pages.json")
        chunks_path = os.path.join(self.chunks_dir, f"{base_name}.jsonl")
        manifest_path = os.path.join(self.manifests_dir, f"{base_name}_manifest.json")

        # load existing pages json if resuming
        pages_json: Dict[str, Any] = {"pages": []}
        if resume_from_checkpoint and os.path.exists(raw_pages_path):
            try:
                with open(raw_pages_path, "r", encoding="utf-8") as f:
                    pages_json = json.load(f)
            except Exception:
                pages_json = {"pages": []}

        start_page_idx = int(checkpoint.get("completed_pages", 0))
        if start_page_idx > 0:
            logging.info(f"Resuming from page {start_page_idx + 1}/{num_pages}")

        # Process pages in batches (demo: batch_size=1)
        for batch_start in range(start_page_idx, num_pages, self.batch_size):
            batch_end = min(batch_start + self.batch_size, num_pages)

            logging.info(f"Processing pages {batch_start + 1}-{batch_end} / {num_pages}")

            batch_images: List[bytes] = []
            page_numbers_1based: List[int] = []

            for page_idx in range(batch_start, batch_end):
                page = doc.load_page(page_idx)
                pix = page.get_pixmap(matrix=fitz.Matrix(self.dpi / 72, self.dpi / 72))
                img_bytes = pix.tobytes("png")
                batch_images.append(img_bytes)

                page_no_1based = page_idx + 1
                page_numbers_1based.append(page_no_1based)

                img_path = os.path.join(pdf_images_dir, f"page_{page_no_1based}.png")
                with open(img_path, "wb") as f:
                    f.write(img_bytes)

            prompt = self._get_extraction_prompt(doc_title, page_numbers_1based[0], page_numbers_1based[-1])
            raw = self._call_openai_with_retry(prompt, batch_images)

            # parse json
            try:
                parsed = json.loads(raw)
            except Exception as e:
                logging.error(f"Failed to parse JSON from model for pages {page_numbers_1based}: {e}")
                parsed = {"pages": []}

            # append/merge pages_json (avoid duplicates)
            existing_pages = {p.get("page"): p for p in pages_json.get("pages", [])}
            for p in parsed.get("pages", []):
                existing_pages[p.get("page")] = p
            pages_json["pages"] = [existing_pages[k] for k in sorted(existing_pages.keys()) if isinstance(k, int)]

            # save raw pages json incrementally
            with open(raw_pages_path, "w", encoding="utf-8") as f:
                json.dump(pages_json, f, ensure_ascii=False, indent=2)

            # update checkpoint
            checkpoint["completed_pages"] = batch_end
            self._save_checkpoint(pdf_path, checkpoint)

        doc.close()

        # build chunks from all pages_json
        chunks = self._build_chunks_from_pages_json(
            doc_id=doc_id,
            file_name=Path(pdf_path).name,
            file_sha=file_sha,
            page_count=num_pages,
            pages_json=pages_json,
            dpi=self.dpi,
            parser_version="demo-v1"
        )

        # write JSONL
        with open(chunks_path, "w", encoding="utf-8") as f:
            for ch in chunks:
                f.write(json.dumps(ch, ensure_ascii=False) + "\n")

        # manifest
        manifest = {
            "doc_id": doc_id,
            "file_name": Path(pdf_path).name,
            "file_sha256": file_sha,
            "page_count": num_pages,
            "dpi": self.dpi,
            "batch_size": self.batch_size,
            "chunks_count": len(chunks),
            "outputs": {
                "raw_pages_json": raw_pages_path,
                "chunks_jsonl": chunks_path,
                "images_dir": pdf_images_dir,
            }
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

        # cleanup checkpoint after success
        cp = self._get_checkpoint_path(pdf_path)
        if os.path.exists(cp):
            os.remove(cp)

        logging.info(f"✅ Done: {Path(pdf_path).name} -> chunks: {len(chunks)}")
        return len(chunks)

    def process_pdfs_in_directory(self, pdf_dir: str, resume_from_checkpoint: bool = True) -> Dict[str, int]:
        pdf_files = glob.glob(os.path.join(pdf_dir, "*.pdf"))
        if not pdf_files:
            logging.warning(f"No PDFs found in: {pdf_dir}")
            return {}
        results: Dict[str, int] = {}
        for i, pdf_path in enumerate(pdf_files, start=1):
            logging.info(f"\n--- Document {i}/{len(pdf_files)}: {Path(pdf_path).name} ---")
            count = self._process_single_pdf(pdf_path, resume_from_checkpoint)
            results[Path(pdf_path).name] = count
        return results


def main():
    INPUT_PDF_DIR = "input_pdfs"
    os.makedirs(INPUT_PDF_DIR, exist_ok=True)

    parser = GenericVisionOCRParser(
        output_dir="ocr_output",
        batch_size=1,     # demo: أعلى دقة
        dpi=500,          # demo
        max_text_chunk_chars=1800,
        overlap_chars=150,
    )

    results = parser.process_pdfs_in_directory(INPUT_PDF_DIR, resume_from_checkpoint=True)

    if results:
        print("\n" + "=" * 60)
        print("✅ Demo OCR -> Header-based Chunks Completed!")
        print(f"Processed {len(results)} PDFs.")
        print("Outputs:")
        print(" - ocr_output/raw_pages_json/   (structured per-page JSON from Vision)")
        print(" - ocr_output/chunks_jsonl/     (READY for RAG ingestion)")
        print(" - ocr_output/manifests/        (doc manifests)")
        print(" - ocr_output/images/           (page images)")
        print("=" * 60)
    else:
        print("\n❌ No PDFs processed. Check logs.")


if __name__ == "__main__":
    main()
