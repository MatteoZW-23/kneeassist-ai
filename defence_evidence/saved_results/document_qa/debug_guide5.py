import importlib.util, traceback
from docx import Document
p=r"C:\Users\HP\Desktop\KneeAssist-AI\scripts\build_guide_aligned_dissertation.py"
s=importlib.util.spec_from_file_location("b",p); b=importlib.util.module_from_spec(s); s.loader.exec_module(b)
def status(doc,label): print(label,any(x.text.strip()=='CHAPTER 2: LITERATURE REVIEW' for x in doc.paragraphs))
origdel=b.delete_block
def dele(doc,a): status(doc,'before del '+a); r=origdel(doc,a); status(doc,'after del '+a); return r
b.delete_block=dele
origrem=b.remove_table_with_headers
def rem(doc,a): status(doc,'before tables');r=origrem(doc,a);status(doc,'after tables');return r
b.remove_table_with_headers=rem
originsert=b.add_inserted_block
def ins(a,items):
 r=originsert(a,items); return r
b.add_inserted_block=ins
origfind=b.find_paragraph
def find(doc,t):
 x=origfind(doc,t); return x
b.find_paragraph=find
d=Document(b.SOURCE)
try: b.guide_content(d)
except Exception: traceback.print_exc()
