import subprocess
from pathlib import Path

from music21 import converter


def abc2midi(abc_path: Path, midi_path: Path):
    """Generate MIDI from .abc"""
    abc = converter.parse(abc_path)
    abc.write("midi", fp=midi_path)


def midi2wav(midi_path: Path):
    """Generate WAV from MIDI"""
    # it doesn't work
    wav_path = "abc.wav"
    subprocess.run(["timidity", midi_path.name, "-Ow", "-o", wav_path], check=True)


def abc2pdf(abc_path: Path):
    """Generate music sheet"""
    abcps: str = "abc.ps"
    sheet: str = "sheet.pdf"
    subprocess.run(["abcm2ps", str(abc_path), "-O", abcps], check=True)
    subprocess.run(["ps2pdf", abcps, sheet], check=True)


def main():
    abc_path: Path = Path("./data") / "sheet_sample.abc"
    midi_path: Path = Path("abc.mid")
    abc2midi(abc_path, midi_path)
    midi2wav(midi_path)
    abc2pdf(abc_path)


if __name__ == "__main__":
    main()
