from lxml import etree
import re

def extract_greek_text(xml_path):
    # Parse XML-filen
    tree = etree.parse(xml_path)
    root = tree.getroot()

    # Håndter navnerum (tjek om det er nødvendigt)
    ns = {'tei': 'http://www.tei-c.org/ns/1.0'}
    
    # Finder alle tekstelementer i dokumentet med tei1.0
    greek_texts = root.xpath('//tei:text//tei:body//tei:p', namespaces=ns)
    # Hvis tei2.0 så: greek_texts = root.xpath('//text//body//p')

    # Ekstraherer teksten fra hvert element
    extracted_text = []
    for elem in greek_texts:
        extracted_text.append(elem.text.strip() if elem.text else "")

    # Samler teksten til en lang string
    greek_output = "\n".join(filter(None, extracted_text))
    
    return greek_output

# Indsæt filsti i anførselstegn
xml_path = ""

# Kald funktionen og print resultatet
greek_output = extract_greek_text(xml_path)
words = greek_output.split()
text = ' '.join(words)

clean_text = re.sub('[0-9a-zA-Z!#€%&/()=?\[\]^_`{|}~«»→-˘¯“”〈〉†‘’]', '', text)

print(text)
