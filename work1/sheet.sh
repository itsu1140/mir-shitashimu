#!/bin/bash

SHEET="sheet_sample"
# generate .wav
abc2midi data/"$SHEET".abc -o abc_mid.midi
timidity abc_mid.midi -Ow -o abc_audio.wav

# generate sheet pdf
abcm2ps data/"$SHEET".abc -O sheet.ps
ps2pdf sheet.ps sheet.pdf
