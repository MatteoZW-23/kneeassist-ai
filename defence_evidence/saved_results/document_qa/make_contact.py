from PIL import Image, ImageDraw
from pathlib import Path
d=Path(r"C:\Users\HP\Desktop\KneeAssist-AI\results\document_qa\extracted_figures")
fs=sorted(d.glob("image*.png"))
thumbs=[]
for f in fs:
    im=Image.open(f).convert("RGB")
    im.thumbnail((1000,600))
    thumbs.append(im.copy())
h=sum(im.height+35 for im in thumbs)
c=Image.new("RGB",(1000,h),"white")
draw=ImageDraw.Draw(c)
y=0
for f,im in zip(fs,thumbs):
    draw.text((10,y+5),f.name,fill="black")
    y+=30
    c.paste(im,(0,y))
    y+=im.height+5
c.save(d / "contact.png")
