from typing import Iterable, List, Optional, TypeVar

from lxml import etree

from utils.parsing._parse import Parser, Document
from utils.text import remove_punctuation

T = TypeVar("T")


def unique(iterable: Iterable[T]) -> List[T]:
    """Turns iterable into the list of its unique elements."""
    return list(set(iterable))


def get_versions(book: etree.Element) -> List[etree.Element]:
    """Returns all versions of the book."""
    tree = etree.ElementTree(book)
    # Matches all version elements
    versions = tree.xpath("version")
    return versions


def _normalize(text: Optional[str]) -> Optional[str]:
    # propagate nones
    if text is None:
        return None
    text = remove_punctuation(text, keep_sentences=False)
    text = text.strip()
    return text


def get_author(version: etree.Element) -> str:
    """Extracts author for a given version of the book"""
    version_author = version.get("author", "")
    version_author = _normalize(version_author)
    return version_author


def get_title(version: etree.Element, book: etree.Element) -> str:
    """Extracts title of the document"""
    book_title = book.get("title", "")
    book_title = _normalize(book_title)
    version_title = version.get("title", "")
    version_title = _normalize(version_title)
    return book_title + " - " + version_title


def get_language(version: etree.Element) -> str:
    """Extracts language from document."""
    version_language = version.get("language")
    version_tree = etree.ElementTree(version)
    # Matches the language attribute of all ms tags
    manuscript_languages = version_tree.xpath("//ms/@language")
    manuscript_languages = unique(manuscript_languages)
    # I checked and there is always only one language, so doing this is fine
    if manuscript_languages:
        manuscript_language = manuscript_languages[0]
    else:
        manuscript_language = ""
    if version_language:
        return version_language
    else:
        return manuscript_language


def get_text(version: etree.Element) -> List[dict]:
    """
    Extracts textual content from a version of the book, producing one entry
    per manuscript where applicable.

    Parameters
    ----------
    version: etree.Element
        The XML element of the version to parse.

    Returns
    ----------
    output: list of dictionaries
        Each dictionary contains manuscript abbreviation and corresponding text.
    """
    version_tree = etree.ElementTree(version)

    # Check if <manuscripts> is defined
    manuscripts = version_tree.xpath("//manuscripts/ms[@show='yes']")
    if manuscripts:
        texts_by_ms = []
        manuscript_keys = {ms.get("abbrev"): [] for ms in manuscripts}

        # Collect readings grouped by manuscript at the unit level
        units = version_tree.xpath("//unit")
        for unit in units:
            unit_readings_by_ms = {key: [] for key in manuscript_keys}
            for reading in unit.xpath("./reading"):
                mss = reading.get("mss", "").strip()  # Manuscripts associated with this reading
                reading_text = reading.text.strip() if reading.text else ""
                for ms_key in mss.split():
                    if ms_key in unit_readings_by_ms:
                        unit_readings_by_ms[ms_key].append(reading_text)
            for ms_key, unit_readings in unit_readings_by_ms.items():
                if unit_readings:
                    manuscript_keys[ms_key].append(" ".join(unit_readings))

        # Create separate outputs for each manuscript
        for ms_key, readings in manuscript_keys.items():
            if readings:
                texts_by_ms.append({
                    "manuscript": ms_key,
                    "text": f"[{ms_key}]\n" + " ".join(readings).strip()
                })
        return texts_by_ms

    else:
        # No manuscripts; return entire text as a single entry
        readings = version_tree.xpath("//reading/text()")
        return [{"manuscript": None, "text": "\n".join(readings)}]


def get_id(version: etree.Element, book: etree.Element) -> str:
    """Extracts file ID from book."""
    book_id = book.get("filename", "")
    version_title = version.get("title", "")
    version_title = _normalize(version_title)
    return book_id + "-" + version_title



class PseudepigraphaParser(Parser):
    def parse_file(self, file: str) -> Iterable[Document]:
        """Parses the file into an iterable of documents, one per manuscript."""
        tree = etree.parse(file)
        book = tree.xpath("/book")[0]
        book_tree = etree.ElementTree(book)
        versions = book_tree.xpath("version")

        for version in versions:
            language = get_language(version)
            if language != "Greek":
                continue

            base_id = get_id(version=version, book=book)
            title = get_title(version=version, book=book)
            author = get_author(version=version)
            texts_by_ms = get_text(version=version)

            for manuscript_entry in texts_by_ms:
                ms_suffix = manuscript_entry["manuscript"]
                ms_suffix = f"-{ms_suffix}" if ms_suffix else ""
                doc_id = f"{base_id}{ms_suffix}"
                doc_title = f"{title} {ms_suffix}" if ms_suffix else title
                yield {
                    "id": doc_id,
                    "title": doc_title,
                    "author": author,
                    "text": manuscript_entry["text"],
                }
