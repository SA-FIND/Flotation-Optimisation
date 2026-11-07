import sys
import docx

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

doc = docx.Document('docs/Ahedor_Flotation_ML_Final_1.0.docx')

print("=== CHECKING OMML NATIVE MATH EQUATIONS ===")
for i, p in enumerate(doc.paragraphs):
    m_texts = p._element.xpath('.//m:t')
    if m_texts:
        m_str = ''.join([mt.text for mt in m_texts if mt.text])
        w_str = p.text.strip()
        print(f"P{i:03d}: [Word Text]: {repr(w_str)}")
        print(f"      [Math OMML]: {repr(m_str)}")
        print()
