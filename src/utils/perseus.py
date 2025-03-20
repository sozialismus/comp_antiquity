from lxml import etree
import re
import string

def clean_text(text: str) -> str:
    """
    Clean the text by removing Latin letters and unwanted punctuation.
    Allowed punctuation for ancient Greek includes: comma, period, semicolon, and the raised dot.
    """
    # Remove Latin letters using a regular expression
    text = re.sub(r"[A-Za-z]", "", text)
    
    # Define allowed punctuation
    allowed_punct = {',', '.', ';', '·'}
    # Create a translation table that maps unwanted punctuation to None
    all_punct = set(string.punctuation)
    disallowed_punct = ''.join(ch for ch in all_punct if ch not in allowed_punct)
    translation_table = str.maketrans('', '', disallowed_punct)
    # Remove unwanted punctuation
    text = text.translate(translation_table)
    
    # Collapse multiple spaces into one
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def extract_greek_text(xml_path: str) -> str:
    """
    Extracts ancient Greek text from TEI/EpiDoc XML files.
    Parses the XML, extracts text from all <p> nodes within the body of the text,
    cleans each text block to remove Latin characters and unwanted punctuation,
    and returns the concatenated result.
    """
    # Parse the XML file
    tree = etree.parse(xml_path)
    root = tree.getroot()
    
    # Define the TEI namespace; EpiDoc files also use this namespace
    ns = {'tei': 'http://www.tei-c.org/ns/1.0'}
    
    # XPath to locate paragraph nodes within the body
    paragraphs = root.xpath('//tei:text//tei:body//tei:p', namespaces=ns)
    
    extracted_text = []
    for p in paragraphs:
        # Use itertext() to extract all text within the <p> element, effectively skipping over <pb> tags
        text = ''.join(p.itertext())
        # Clean the text to remove Latin letters and disallowed punctuation
        cleaned_text = clean_text(text)
        if cleaned_text:
            extracted_text.append(cleaned_text)
    
    # Join paragraphs into a single string with line breaks
    return "\n".join(extracted_text)

# Example usage:
xml_path = "/home/gnosis/Documents/au_work/main/corpora/Plutarch/tlg0561.tlg001.perseus-grc2.xml"
greek_output = extract_greek_text(xml_path)
print(greek_output)
