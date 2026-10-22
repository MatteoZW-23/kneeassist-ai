"""Create visual-QA contact sheets from rendered PDF page PNGs."""
from pathlib import Path
import sys
from PIL import Image, ImageDraw


def main() -> None:
    source = Path(sys.argv[1])
    pages = sorted(source.glob("*.png"))
    if not pages:
        raise SystemExit(f"No PNG pages in {source}")
    for old in source.glob("contact-sheet-*.png"):
        old.unlink()
    for start in range(0, len(pages), 9):
        items = pages[start:start + 9]
        sheet = Image.new("RGB", (930, 1310), "#d9e4f2")
        for i, item in enumerate(items):
            image = Image.open(item).convert("RGB")
            image.thumbnail((290, 400))
            tile = Image.new("RGB", (310, 435), "white")
            tile.paste(image, ((310 - image.width) // 2, 25))
            ImageDraw.Draw(tile).text((10, 5), f"Page {int(item.stem.split('-')[-1])}", fill="black")
            sheet.paste(tile, ((i % 3) * 310, (i // 3) * 435))
        sheet.save(source / f"contact-sheet-{start // 9 + 1}.png")
    print(f"Created {((len(pages) - 1) // 9) + 1} contact sheets for {len(pages)} pages")


if __name__ == "__main__":
    main()
