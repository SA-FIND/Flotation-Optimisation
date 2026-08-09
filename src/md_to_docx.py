import os
from docx import Document
from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH

def markdown_to_docx(md_path, docx_path):
    doc = Document()
    
    # Set default font
    style = doc.styles['Normal']
    font = style.font
    font.name = 'Times New Roman'
    font.size = Pt(11)
    
    with open(md_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if line.startswith('# '):
            p = doc.add_heading(line[2:], 0)
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        elif line.startswith('## '):
            doc.add_heading(line[3:], 1)
        elif line.startswith('### '):
            doc.add_heading(line[4:], 2)
        elif line.startswith('#### '):
            doc.add_heading(line[5:], 3)
        elif line.startswith('* '):
            p = doc.add_paragraph(line[2:], style='List Bullet')
        elif line.startswith('1. ') or line.startswith('2. ') or line.startswith('3. ') or line.startswith('4. ') or line.startswith('5. ') or line.startswith('6. '):
            p = doc.add_paragraph(line[3:], style='List Number')
        elif line.startswith('!['):
            import re
            match = re.match(r'!\[.*?\]\((.*?)\)', line)
            if match:
                img_path = match.group(1)
                if img_path.startswith('/C:'):
                    img_path = img_path[1:]
                try:
                    doc.add_picture(img_path, width=Inches(6.0))
                except Exception as e:
                    print(f"Failed to add image {img_path}: {e}")
        else:
            doc.add_paragraph(line)
            
    doc.save(docx_path)
    print(f"Successfully saved to {docx_path}")

import sys
from pathlib import Path

if __name__ == '__main__':
    if len(sys.argv) > 2:
        md_path = sys.argv[1]
        docx_path = sys.argv[2]
    else:
        # Default behavior: convert the rewritten_paper artifact if it exists, save to docs/
        PROJECT_ROOT = Path(__file__).resolve().parent.parent
        md_path = str(PROJECT_ROOT / 'docs' / 'Ahedor_Flotation_ML_Final_1.1.md')
        docx_path = str(PROJECT_ROOT / 'docs' / 'Ahedor_Flotation_ML_Final_1.1.docx')
        
    markdown_to_docx(md_path, docx_path)
