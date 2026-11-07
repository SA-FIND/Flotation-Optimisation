import sys
import docx
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

doc = docx.Document('docs/Ahedor_Flotation_ML_Final_1.0.docx')

print("=== PARAGRAPHS 28 TO 115 ===")
for i, p in enumerate(doc.paragraphs):
    if (28 <= i <= 85) or (105 <= i <= 115):
        print(f"P{i:03d} [{p.style.name}]: {p.text}")

print("\n=== CHECKING FIGURES IN DOCX ===")
images = 0
for rel in doc.part.rels.values():
    if "image" in rel.target_ref:
        images += 1
print(f"Total embedded images found in document: {images}")
