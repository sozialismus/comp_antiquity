
from typing import Tuple

from lxml import etree

from utils.parsing._parse import Parser, Document


def get_title(tree: etree.ElementTree) -> str:
    """
    Extracts the title of the document from the <title> element.
    """
    title_list = tree.xpath("//book/title/text()")
    if title_list:
        return title_list[0].strip()
    return ""


def get_id(path: str) -> str:
    """Get SBLGNT ID of file given its path."""
    file_name = path.split("/")[-1]
    sblgnt_id = file_name[: file_name.find(".xml")]
    return sblgnt_id


def get_text(tree: etree.ElementTree) -> str:
    """Extracts text by processing each <p> element within the <book> element.
    
    For each paragraph, concatenates the text from <w> and <suffix> elements,
    ignoring <verse-number> elements. Each <w> element is followed by a space,
    while <suffix> text is appended directly. Paragraphs are joined with newline
    characters.
    """
    paragraphs = tree.xpath("//book/p")
    paragraph_texts = []
    
    for p in paragraphs:
        parts = []
        # Process each child element in order.
        for child in p:
            if child.tag == "verse-number":
                # Skip verse numbers entirely.
                continue
            elif child.tag == "w":
                # Append word with a trailing space.
                if child.text:
                    parts.append(child.text.strip() + " ")
            elif child.tag == "suffix":
                # Append suffix text without additional space.
                if child.text:
                    parts.append(child.text.strip())
            else:
                # In case of any other elements, add their text with a trailing space.
                if child.text:
                    parts.append(child.text.strip() + " ")
        # Join the parts to form the paragraph text.
        paragraph_text = "".join(parts).strip()
        if paragraph_text:
            paragraph_texts.append(paragraph_text)
    
    # Join all paragraph texts with a newline separator.
    return "\n".join(paragraph_texts)


class SBLGNTParser(Parser):
    """Parser for SBLGNT files"""

    def parse_file(self, file: str) -> Tuple[Document]:
        """Parses file into a document"""
        tree = etree.parse(file)
        doc: Document = {
            "id": get_id(path=file),
            "title": get_title(tree=tree),
            "author": "",
            "text": get_text(tree=tree),
        }
        return (doc,)
