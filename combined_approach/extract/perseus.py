import xml.etree.ElementTree as ET
import os
import argparse

# Namespace mapping for TEI
NS = {"tei": "http://www.tei-c.org/ns/1.0"}

# Set of fully qualified tag names to ignore during text extraction
IGNORE_TAGS = {
    f"{{{NS['tei']}}}note",
    f"{{{NS['tei']}}}pb",
    f"{{{NS['tei']}}}milestone"
}

def get_clean_text(element, ignore_tags=IGNORE_TAGS):
    """
    Recursively extracts text from an XML element while skipping any child elements
    whose tags are in the ignore_tags set.
    
    Args:
        element (ET.Element): The XML element to extract text from.
        ignore_tags (set): A set of tag names (with namespaces) to skip.
        
    Returns:
        str: The concatenated text content of the element.
    """
    texts = []
    # Only include text from this element if its tag is not in ignore_tags.
    if element.tag not in ignore_tags:
        if element.text:
            texts.append(element.text)
        for child in element:
            texts.append(get_clean_text(child, ignore_tags))
            if child.tail:
                texts.append(child.tail)
    return "".join(texts).strip()

def extract_text_from_tei(xml_file_path):
    """
    Extracts the main Ancient Greek text from a TEI XML file.
    
    This function is tailored to TEI files structured like your example:
      - It uses namespace-aware searching.
      - It looks inside <text><body> for a <div type="edition"> element.
      - Within that, chapters are identified as <div type="textpart" subtype="chapter">,
        and sections as <div type="textpart" subtype="section">.
      - It extracts chapter and section titles (if available) and the text content from <p> elements.
      - Supplementary tags such as <note>, <pb>, and <milestone> are ignored.
    
    Args:
        xml_file_path (str): Path to the TEI XML file.
        
    Returns:
        str: A string containing the extracted text with structure.
    """
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    
    # Locate the text body using namespace-aware queries.
    text_elem = root.find('.//tei:text', NS)
    if text_elem is None:
        text_elem = root  # fallback in case structure is different

    body = text_elem.find('.//tei:body', NS)
    if body is None:
        body = root

    # In many TEI files the main text is within a <div type="edition">.
    edition = body.find('tei:div[@type="edition"]', NS)
    if edition is None:
        edition = body

    output_lines = []

    # Optionally, include the work title if present.
    work_head = edition.find('tei:head', NS)
    if work_head is not None:
        work_title = get_clean_text(work_head)
        output_lines.append(f"Work Title: {work_title}")
        output_lines.append("")  # blank line for separation

    # Find chapters: <div type="textpart" subtype="chapter">
    chapters = edition.findall('tei:div[@type="textpart"][@subtype="chapter"]', NS)
    if chapters:
        for chapter in chapters:
            chap_num = chapter.get("n", "Unknown")
            # If a chapter head is provided, use it; otherwise, default to chapter number.
            chap_head_elem = chapter.find('tei:head', NS)
            chap_title = get_clean_text(chap_head_elem) if chap_head_elem is not None else f"Chapter {chap_num}"
            output_lines.append(f"{chap_title}")
            
            # Find sections within the chapter: <div type="textpart" subtype="section">
            sections = chapter.findall('tei:div[@type="textpart"][@subtype="section"]', NS)
            if sections:
                for section in sections:
                    sec_num = section.get("n", "Unknown")
                    sec_head_elem = section.find('tei:head', NS)
                    sec_title = get_clean_text(sec_head_elem) if sec_head_elem is not None else f"Section {sec_num}"
                    output_lines.append(f"  {sec_title}")
                    
                    # Extract all <p> elements within the section.
                    paragraphs = section.findall('.//tei:p', NS)
                    for para in paragraphs:
                        para_text = get_clean_text(para)
                        if para_text:
                            output_lines.append(f"    {para_text}")
                    output_lines.append("")  # blank line between sections
            else:
                # Fallback: if no sections, extract <p> elements directly from the chapter.
                paragraphs = chapter.findall('.//tei:p', NS)
                for para in paragraphs:
                    para_text = get_clean_text(para)
                    if para_text:
                        output_lines.append(f"  {para_text}")
                output_lines.append("")  # blank line after chapter
            output_lines.append("")  # blank line between chapters
    else:
        # If no chapters found, extract all paragraphs from the body.
        paragraphs = body.findall('.//tei:p', NS)
        for para in paragraphs:
            para_text = get_clean_text(para)
            if para_text:
                output_lines.append(para_text)

    return "\n".join(output_lines)

def process_directory(input_dir, output_dir):
    """
    Processes all TEI XML files in the input directory, extracts the Ancient Greek text,
    and writes the results to corresponding text files in the output directory.
    
    Args:
        input_dir (str): Path to the directory containing TEI XML files.
        output_dir (str): Path to the directory where output text files will be saved.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith('.xml'):
            input_file_path = os.path.join(input_dir, filename)
            try:
                extracted_text = extract_text_from_tei(input_file_path)
            except Exception as e:
                print(f"Error processing file {input_file_path}: {e}")
                continue

            output_filename = os.path.splitext(filename)[0] + '.txt'
            output_file_path = os.path.join(output_dir, output_filename)
            with open(output_file_path, 'w', encoding='utf-8') as f:
                f.write(extracted_text)
            print(f"Processed {filename} -> {output_filename}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Extract Ancient Greek text from TEI XML files in a directory"
    )
    parser.add_argument("input_dir", help="Path to the input directory containing TEI XML files")
    parser.add_argument("output_dir", help="Path to the output directory to save extracted text files")
    args = parser.parse_args()
    
    process_directory(args.input_dir, args.output_dir)
