import os
import json
import struct
import collections
import re
import spacy

nlp = spacy.load('en', disable=['tagger', 'ner'], vectors=False)
print('Spacy loaded')


def get_tokens(doc):  # 分词用
    doc = nlp(doc)
    new_tokens = []
    for k in doc:
        new_tokens.append(k.text)
    return new_tokens


def process_tokens(st):
    return " ".join(st)


multi_ref = json.load(open('multi_reference_test.json', 'r', encoding='utf-8'))

for example_id, value in multi_ref.items():
    temp_res = []
    for response in value["responses"]:
        response = response.lower()
        modified_response = process_tokens(get_tokens(response))
        temp_res.append(modified_response)
    value["responses"] = temp_res

    temp_span = []
    for span in value["spans"]:
        if isinstance(span, int):
            span = str(span)
        try:
            span = span.lower()
            modified_span = process_tokens(get_tokens(span))
            temp_span.append(modified_span)
        except TypeError as e:
            print(value["spans"])
            print(span)
    value["spans"] = temp_span

with open(os.path.join("modified_multi_reference_test.json"), 'w', encoding='utf-8') as w:
    json.dump(multi_ref, w)
