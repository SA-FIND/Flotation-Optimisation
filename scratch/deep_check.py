import sys
import docx
import re
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

doc_path = Path('docs/Ahedor_Flotation_ML_Final_1.0.docx')
doc = docx.Document(str(doc_path))

print(f"=== MANUSCRIPT AUDIT: {doc_path.name} ===")
print(f"File Size: {doc_path.stat().st_size / 1024:.1f} KB")
print(f"Total Paragraphs: {len(doc.paragraphs)}")
print(f"Total Tables: {len(doc.tables)}")

print("\n--- 1. PARAGRAPH-BY-PARAGRAPH DUMP ---")
for i, p in enumerate(doc.paragraphs):
    txt = p.text.strip()
    if txt:
        print(f"[{i:03d}|{p.style.name}]: {txt}")

print("\n--- 2. ALL TABLES DUMP ---")
for t_idx, table in enumerate(doc.tables):
    print(f"\n[TABLE {t_idx+1}] ({len(table.rows)} rows x {len(table.columns)} cols)")
    for r_idx, row in enumerate(table.rows):
        cells = [c.text.replace('\n', ' ').strip() for c in row.cells]
        print(f"  R{r_idx}: {' | '.join(cells)}")

print("\n--- 3. IMAGES & FIGURES AUDIT ---")
images = []
for rel_id, rel in doc.part.rels.items():
    if "image" in rel.target_ref:
        images.append((rel_id, rel.target_ref))
print(f"Total embedded images: {len(images)}")
for r_id, ref in images:
    print(f"  Image: {r_id} -> {ref}")
