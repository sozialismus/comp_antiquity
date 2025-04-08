from typing import Iterable, List, Optional, TypeVar
from pathlib import Path
from utils.parsing._parse import Parser, Document
from utils.text import remove_punctuation

class AttalusScrapedTextParser(Parser):
    def parse_file(self, file: str) -> Iterable[Document]:
        """Parses plain text files into documents."""
        with open(file, encoding="utf-8") as f:
            text = f.read()
        # Example: Use filename as ID and generate a default title
        doc_id = Path(file).stem
        yield { #hardcoded values, since there's only one texts, and without dynamic references to fetch metadata from
            "id": doc_id,
            "title": "Alexander Romance",
            "author": "Pseudo-Callisthenes",
            "text": text.strip(),
        }
