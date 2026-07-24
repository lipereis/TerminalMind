"""Generate docs/demo.gif — run: python scripts/make_demo_gif.py"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "demo.gif"

FRAMES: list[list[str]] = [
    [
        "$ terminalmind ingest ./notes/",
        "",
        "Ingested notes/intro.md (412 chars)",
        "Ingested notes/claims.md (880 chars)",
        "Done: 2 ingested, 0 skipped (2 files)",
    ],
    [
        "$ terminalmind search \"What are the main claims?\"",
        "",
        "┌──────── Summary ────────┐",
        "│ Notes argue RAG-lite    │",
        "│ beats keyword noise.    │",
        "└─────────────────────────┘",
    ],
    [
        "Key Points",
        " • Ingest folders of .md/.txt",
        " • Structured Pydantic output",
        " • Sources cite matched chunks",
        "",
        "Sources",
        " • `abc:0` — Notes argue RAG-lite…",
    ],
    [
        "$ terminalmind history --export out.md",
        "",
        "Exported 1 sessions → out.md",
        "",
        "$ terminalmind chat",
        "you> _",
    ],
]


def _font() -> ImageFont.ImageFont:
    for name in ("consola.ttf", "Consolas.ttf", "DejaVuSansMono.ttf", "cour.ttf"):
        try:
            return ImageFont.truetype(name, 16)
        except OSError:
            continue
    return ImageFont.load_default()


def _render(lines: list[str], font: ImageFont.ImageFont) -> Image.Image:
    width, height = 720, 280
    img = Image.new("RGB", (width, height), "#0d1117")
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, width, 28), fill="#161b22")
    draw.text((12, 6), "TerminalMind", fill="#58a6ff", font=font)
    y = 44
    for line in lines:
        color = "#c9d1d9"
        if line.startswith("$"):
            color = "#3fb950"
        elif line.startswith("┌") or line.startswith("│") or line.startswith("└"):
            color = "#79c0ff"
        elif line.startswith(" •") or line.startswith("Sources") or line.startswith("Key"):
            color = "#e3b341"
        draw.text((16, y), line, fill=color, font=font)
        y += 20
    return img


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    font = _font()
    frames = [_render(lines, font) for lines in FRAMES]
    frames[0].save(
        OUT,
        save_all=True,
        append_images=frames[1:],
        duration=1600,
        loop=0,
        optimize=True,
    )
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
