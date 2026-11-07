import sys
import re
import docx
import pandas as pd
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

doc = docx.Document('docs/Ahedor_Flotation_ML_Final_1.0.docx')

print("=== 1. EXTRACTING IN-TEXT CITATIONS ===")
full_text = "\n".join([p.text for p in doc.paragraphs])

# Find citation patterns like (Author, Year) or Author (Year)
in_text_citations = re.findall(r'([A-Z][a-zA-Z\s\-]+(?:\set\sal\.)?),?\s\(([12][90]\d\d[a-z]?)\)', full_text)
in_text_citations += re.findall(r'\(([A-Z][a-zA-Z\s\-]+(?:\set\sal\.)?),?\s([12][90]\d\d[a-z]?)\)', full_text)

print(f"Total raw in-text citation matches: {len(in_text_citations)}")
unique_in_text = sorted(list(set([(a.strip(), y.strip()) for a, y in in_text_citations])))
for a, y in unique_in_text:
    print(f"  In-text: {a} ({y})")

print("\n=== 2. EXTRACTING BIBLIOGRAPHY ENTRIES ===")
bib_entries = []
in_bib = False
for p in doc.paragraphs:
    if p.text.strip() == "References":
        in_bib = True
        continue
    if in_bib and p.text.strip():
        bib_entries.append(p.text.strip())

print(f"Total bibliography entries: {len(bib_entries)}")
for i, b in enumerate(bib_entries):
    first_author = b.split('.')[0].split(',')[0].strip()
    # extract year
    year_match = re.search(r'\b(19\d\d|20\d\d)\b', b)
    year = year_match.group(1) if year_match else "????"
    print(f"  [{i+1:02d}] {first_author} ({year}): {b[:60]}...")

print("\n=== 3. CHECKING TABLE DATA CONSISTENCY ===")
for t_idx, table in enumerate(doc.tables):
    print(f"\n--- Table {t_idx+1} ---")
    for r_idx, row in enumerate(table.rows):
        cells = [c.text.replace('\n', ' ').strip() for c in row.cells]
        print(f"  Row {r_idx}: {' | '.join(cells)}")
