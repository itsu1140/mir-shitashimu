import subprocess
from pathlib import Path

from music21 import converter


def abc2midi(abc_path: Path):
    abc = converter.parse(abc_path)
    abc.write("midi", fp="abc_midi.mid")


def abc2pdf(abc_path: Path):
    abcps: str = "abc.ps"
    sheet: str = "sheet.pdf"
    subprocess.run(["abcm2ps", str(abc_path), "-O", abcps], check=True)
    subprocess.run(["ps2pdf", abcps, sheet], check=True)


def main():
    abc_path: Path = Path("./data") / "sheet_sample.abc"
    abc2midi(abc_path)
    abc2pdf(abc_path)


if __name__ == "__main__":
    main()
