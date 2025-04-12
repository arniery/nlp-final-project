# %%
import nltk, re, pprint
from nltk import word_tokenize, sent_tokenize
import os
import json
import xml.etree.ElementTree as ET
from nltk.corpus.reader.api import CorpusReader
from nltk.corpus.reader.util import find_corpus_fileids


# %%
class CustomCorpusReader(CorpusReader):
    def __init__(self, root):
        super().__init__(root, fileids=None)
        self.annotations_dir = os.path.join(root, 'annotations')
        self.cases_dir = os.path.join(root, 'cases')

    def annotations(self):
        annotations = {}
        for fname in os.listdir(self.annotations_dir):
            if fname.endswith('.json'):
                fpath = os.path.join(self.annotations_dir, fname)
                with open(fpath, 'r', encoding='utf-8') as f:
                    annotations[fname] = json.load(f)
        return annotations

    def case_files(self):
        """Recursively walk through group/year folders and parse XML files."""
        cases = {}
        for root, _, files in os.walk(self.cases_dir):
            for file in files:
                if file.endswith('.xml'):
                    full_path = os.path.join(root, file)
                    try:
                        tree = ET.parse(full_path)
                        root_elem = tree.getroot()

                        # optional: create a case ID from its path
                        rel_path = os.path.relpath(full_path, self.cases_dir)
                        case_id = rel_path.replace(os.sep, '_').replace('.xml', '')

                        cases[case_id] = root_elem
                    except ET.ParseError as e:
                        print(f"Error parsing {full_path}: {e}")
        return cases
    
    def extract_text(self, xml_root):
        """Extract all text from an XML tree, ignoring tags."""
        return ' '.join(xml_root.itertext())

    def case_text_stats(self):
        """
        Return a dict of sentence and word counts per case.
        Structure: { case_id: {'sentences': x, 'words': y} }
        """
        stats = {}
        case_xml = self.case_files()
        
        for case_id, root in case_xml.items():
            raw_text = self.extract_text(root)
            sentences = sent_tokenize(raw_text)
            words = word_tokenize(raw_text)
            
            stats[case_id] = {
                'sentences': len(sentences),
                'words': len(words)
            }

        return stats

# %%
reader_clc = CustomCorpusReader('/Users/arniexx/Desktop/datasets & nlp/nlp final project/CLCmini/corpus')
stats_clc = reader_clc.case_text_stats()

# print all sentence and word counts, 15 cases/xml files
print("Case Text Statistics:\n")
for case_id, counts in sorted(stats_clc.items()):
    print(f"{case_id}: {counts['sentences']} sentences, {counts['words']} words")

# %%
def clean_text(text):
    # remove special characters (keep basic punctuation)
    text = re.sub(r"[^a-zA-Z0-9\s.,;:!?'\"]+", '', text)
    
    # normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def export_combined_txt(reader, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        for case_id, xml_root in reader.case_files().items():
            text = reader.extract_text(xml_root)
            cleaned = clean_text(text)
            f.write(f"--- {case_id} ---\n")
            f.write(cleaned + "\n\n")

export_combined_txt(reader_clc, '/Users/arniexx/Desktop/datasets & nlp/nlp final project/corpus_combined.txt')

# %%
import spacy
nlp = spacy.load("en_core_web_sm")
print(nlp.pipe_names)

# %%
# read the corpus
with open("corpus_combined.txt", "r", encoding="utf-8") as file:
    text = file.read()

# process the text
doc = nlp(text)

# iterate over sentences and print POS-tagged tokens
for sent in doc.sents:
    print(f"\nSentence: {sent.text}")
    for token in sent:
        print(f"{token.text}\t{token.pos_}\t{token.tag_}\t{token.dep_}")

# %%
nlp2 = spacy.load("en_core_web_md")
anchor_phrases = [
    "in accordance with",
    "on behalf of",
    "in respect of",
    "by virtue of",
    "in relation to",
    "with regard to",
    "as a result of"
]

anchor_docs = [nlp2(p) for p in anchor_phrases]

# %%
from itertools import islice

def get_candidate_phrases(doc, min_len=3, max_len=4):
    candidates = []
    for i in range(len(doc)):
        for size in range(min_len, max_len + 1):
            span = doc[i:i+size]
            if len(span) < size:
                continue
            if any(token.pos_ == "ADP" for token in span):
                candidates.append(span)
    return candidates

# %%
# load the text
with open("corpus_combined.txt", "r", encoding="utf-8") as f:
    text = f.read()

doc = nlp2(text)
candidates = get_candidate_phrases(doc)

similar_matches = []

# compare candidates to anchors
for span in candidates:
    for anchor_doc in anchor_docs:
        similarity = span.similarity(anchor_doc)
        if similarity > 0.85:  # Tune this threshold
            similar_matches.append((span.text, similarity))

for phrase, score in sorted(similar_matches, key=lambda x: -x[1]):
    print(f"matched: '{phrase}' (similarity: {score:.2f})")


