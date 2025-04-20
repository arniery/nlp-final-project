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
        # recursively walk through group/year folders and parse XML files.
        cases = {}
        for root, _, files in os.walk(self.cases_dir):
            for file in files:
                if file.endswith('.xml'):
                    full_path = os.path.join(root, file)
                    try:
                        tree = ET.parse(full_path)
                        root_elem = tree.getroot()

                        # create a case ID from its path
                        rel_path = os.path.relpath(full_path, self.cases_dir)
                        case_id = rel_path.replace(os.sep, '_').replace('.xml', '')

                        cases[case_id] = root_elem
                    except ET.ParseError as e:
                        print(f"error parsing {full_path}: {e}")
        return cases
    
    def extract_text(self, xml_root):
        # extract all text from an .xml tree, ignoring tags.
        return ' '.join(xml_root.itertext())

    def case_text_stats(self):
        # return a list of sentence and word counts per case.
        
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
with open("corpus_combined.txt", "r", encoding="utf-8") as file:
    text = file.read()
print(text[:500])  # Print first 500 characters

# %%
import spacy
nlp = spacy.load("en_core_web_sm")
print(nlp.pipe_names)

# read the corpus
with open("corpus_combined.txt", "r", encoding="utf-8") as file:
    text = file.read()

# process the text
doc = nlp(text)

# %%
### SEMANTIC MATCHER
# returns candidates that start with a prep, contain at least 2 preps, no proper nouns, no named entities
# compares candidates semantically to a list of known complex preps
# returns ranked list of matches with semantic context

from itertools import islice

import spacy

def find_semantic_prepositions(
    text,
    anchor_phrases=None,
    min_len=2,
    max_len=4,
    similarity_threshold=0.85,
    exclude_propn=True,
    exclude_named_ents=True
):

    nlp2 = spacy.load("en_core_web_md")
    anchor_phrases = [
        "in accordance with",
        "on behalf of",
        "in respect of",
        "by virtue of",
        "in relation to",
        "with regard to",
        "as a result of",
        "on the basis of",
        "in conformity with",
        "in spite of"
    ]

    anchor_docs = [nlp2(p) for p in anchor_phrases]
    results = []

    # phrase extraction
    for sent in doc.sents:
        for i in range(len(sent)):
            for size in range(min_len, max_len + 1):
                span = sent[i:i+size]
                if len(span) < size:
                    continue

                # must start with a preposition
                if span[0].pos_ != "ADP":
                    continue
                # must contain at least 2 prepositions
                if sum(1 for token in span if token.pos_ == "ADP") < 2:
                    continue
                # exclude proper nouns
                if exclude_propn and any(token.pos_ == "PROPN" for token in span):
                    continue
                # exclude named entities
                if exclude_named_ents and any(token.ent_type_ != "" for token in span):
                    continue

                # compare with anchors
                for anchor in anchor_docs:
                    sim = span.similarity(anchor)
                    if sim >= similarity_threshold:
                        results.append({
                            "phrase": span.text,
                            "similarity": sim,
                            "sentence": sent.text.strip()
                        })
    # sort by similarity
    results = sorted(results, key=lambda x: -x["similarity"])
    return results

# %%
with open("corpus_combined.txt", "r", encoding="utf-8") as f:
    corpus = f.read()

matches = find_semantic_prepositions(corpus)

for match in matches:
    print(f"Matched: '{match['phrase']}' (sim={match['similarity']:.2f})")
    print(f" → Sentence: {match['sentence']}\n")

# %%
for sent in doc.sents:
    print(f"Sentence: {sent.text}")

# %%
# find the most common complex props from the list above
from collections import Counter

def get_common_complex_preps(
    text,
    min_len=2,
    max_len=4,
    min_preps=2,
    exclude_propn=True,
    exclude_named_ents=True,
    exclude_punct=True,
    max_noun_ratio=0.5,
    top_n=20     
):
    nlp2 = spacy.load("en_core_web_md")
    doc = nlp2(text)
    phrase_counter = Counter()
        
    for sent in doc.sents:
        for i in range(len(sent)):
            for size in range(min_len, max_len + 1):
                span = sent[i:i+size]
                if len(span) < size:
                    continue
                
                tokens = list(span)
                span_text = span.text.strip()
                
                # must start with a preposition
                if tokens[0].pos_ != "ADP":
                    continue
                # must contain at least min_preps ADPs
                if sum(1 for tok in tokens if tok.pos_ == "ADP") < min_preps:
                    continue
                # exclude if mostly punctuation or ends in punct
                if any(tok.is_punct for tok in tokens) or tokens[-1].is_punct:
                    continue
                # skip spans with non-alphabetic tokens (like "law.")
                if not all(tok.is_alpha for tok in tokens):
                    continue
                # exclude proper nouns
                if exclude_propn and any(tok.pos_ == "PROPN" for tok in tokens):
                    continue
                # exclude named entities
                if exclude_named_ents and any(tok.ent_type_ != "" for tok in tokens):
                    continue
                # ignore noun phrases like "the court of appeal"
                if all(tok.pos_ in {"DET", "NOUN", "ADP"} for tok in tokens):
                    noun_count = sum(1 for tok in tokens if tok.pos_ == "NOUN")
                    if noun_count / len(tokens) > 0.5:
                        continue

                phrase = span.text.lower().strip()
                phrase_counter[phrase] += 1

    return phrase_counter.most_common(top_n)

# %%
with open("corpus_combined.txt", "r", encoding="utf-8") as f:
    corpus = f.read()

top_phrases = get_common_complex_preps(corpus, top_n=30)

print("Common Complex Prepositional Phrases:\n")
for phrase, freq in top_phrases:
    print(f"• {phrase} — {freq} occurrences")

# %%
phrase_map = {
    "as to": "about",
    "in relation to": "about",
    "in respect of": "about",
    "out of": "from",
    "for the purposes of": "for",
    "in accordance with": "according to",
    "in the case of": "in",
    "in the context of": "when",
    "from time to time": "sometimes",
    "for the purpose of": "for",
    "on behalf of": "for",
    "in the light of": "considering",
    "by reason of": "because",
    "as a result of": "because",
    "by way of": "for",
    "on grounds of": "because of",
    "as a matter of": "about",
    "on the part of": "for",
    "as in": "meaning",
    "on the ground of": "because of",
    "in the course of": "during",
    "by virtue of": "because of"
}

# %%
def extract_training_pairs(text, phrase_map):
    doc = nlp(text)
    results = []

    for sent in doc.sents:
        sent_text = sent.text.lower()
        for complex_pp, simple_pp in phrase_map.items():
            if complex_pp in sent_text:
                # only match full token spans, not substrings inside words
                start_idx = sent_text.find(complex_pp)
                end_idx = start_idx + len(complex_pp)
                results.append({
                    "sentence": sent.text.strip(),
                    "complex": complex_pp,
                    "simple": simple_pp
                })

    return results

# %%
# reformatting my dataset for model training
import csv

input_path = "training_pairs.csv"
output_path = "reformatted_dataset.csv"

with open(input_path, newline='', encoding='utf-8') as infile, \
     open(output_path, "w", newline='', encoding='utf-8') as outfile:
    
    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    # write header
    writer.writerow(["input", "target"])

    for row in reader:
        sentence, complex_phrase, simplified_phrase = row
        input_text = f'Simplify "{complex_phrase}" in: {sentence.strip()}'
        output_text = simplified_phrase.strip()
        writer.writerow([input_text, output_text])


# %%
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
from datasets import load_dataset
import torch
torch.mps.empty_cache()

# %%
# check MPS; was having problems with model using fp16 on my mac m1
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# paths & model
dataset_path = "reformatted_dataset.csv"
model_name = "google/flan-t5-base"

# load dataset
dataset = load_dataset("csv", data_files=dataset_path)
dataset["train"] = dataset["train"].select(range(1000))  # for quick testing

# tokenizer & model
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)
model.to(device)
model.gradient_checkpointing_enable()

# preprocessing
def preprocess(example):
    inputs = tokenizer(
        example["input"],
        max_length=256,
        padding="max_length",
        truncation=True
    )
    labels = tokenizer(
        example["target"],
        max_length=32,
        padding="max_length",
        truncation=True
    )
    inputs["labels"] = labels["input_ids"]
    return inputs

tokenized = dataset.map(preprocess, batched=True)

# data collator
collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# training args
training_args = TrainingArguments(
    output_dir="./flan-t5-span-simplifier",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=1.0,
    max_steps=300,
    logging_steps=10,
    save_strategy="no",
    report_to="none",
    fp16=False,    # force off
    bf16=False     # force off
)

# trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    tokenizer=tokenizer,
    data_collator=collator,
)

# train
trainer.train()

# save final model
trainer.save_model("./flan-t5-span-simplifier-final")

# %%
# test the trained model

# load the model and tokenizer
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

model_path = "./flan-t5-span-simplifier-final"
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)

# %%
input_text = (
    'Simplify the phrase "in the case of" in the sentence: '
    'I would point out that, in the case of a consumer who was also a purchaser, '
    'no contract and no warranty of fitness would apply.'
)

inputs = tokenizer(input_text, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=10,
        num_beams=4,
        early_stopping=True
    )

prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Prediction:", prediction)

# %%
# trying to output the whole sentence with replacement

original_phrase = "in the course of"

# original sentence
full_sentence = (
   "In any event, as matters had developed in the course of the trial since April 2017, it is apparent from Mr. Edgell's evidence that SWP continues to review events against the section 1491 criteria."

)

# build prompt for model
prompt = f'Simplify "{original_phrase}" in: {full_sentence}'

# tokenize and generate
inputs = tokenizer(prompt, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=10,
        num_beams=4,
        early_stopping=True
    )
simplified_phrase = tokenizer.decode(outputs[0], skip_special_tokens=True)

# reconstruct sentence
simplified_sentence = full_sentence.replace(original_phrase, simplified_phrase)

# output
print("Original Phrase: ", original_phrase)
print("Simplified Phrase: ", simplified_phrase)
print("Full Sentence:")
print(simplified_sentence)


