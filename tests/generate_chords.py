#!/usr/bin/env python3
"""
Generate bass/guitar chord samples (B0–E6) via FluidSynth → OGG,
scale into [-1,1], normalize to 95% peak.
"""

import os
import numpy as np
import soundfile as sf
import librosa
import fluidsynth

# 1. Configuration
MIN_MIDI, MAX_MIDI = 35, 100 # B0-E6
SR = 44100
DUR, DECAY = 1.0, 1.0
VEL = 100
SF2 = '/usr/share/soundfonts/FluidR3_GM.sf2'
OUT = "chord-samples"

CHORD_TEMPLATES = {
    "maj":  [0, 4, 7],
    "min":  [0, 3, 7],
    "power":[0, 7],
    "7":    [0, 4, 7, 10],
    "maj7": [0, 4, 7, 11],
    "m7":   [0, 3, 7, 10],
    "dim":  [0, 3, 6],
    "aug":  [0, 4, 8],
    "sus2": [0, 2, 7],
    "sus4": [0, 5, 7],
}

os.makedirs(OUT, exist_ok=True)

PITCH_NAMES = "C C# D D# E F F# G G# A A# B".split()

def note_name(n:int)->str:
    return f"{PITCH_NAMES[n%12]}{n//12-2}"

def generate_chord_samples(root:int, ivals:list[int]) -> np.ndarray:
    fl = fluidsynth.Synth(samplerate=SR, audio_driver="null")
    fl.setting("synth.gain", "0.2")           # optional master-gain reduction
    sfid = fl.sfload(SF2)
    fl.program_select(0, sfid, 0, 0)

    for i in ivals: fl.noteon(0, root+i, VEL)
    s1 = np.array(fl.get_samples(int(SR*DUR)), dtype=np.float32)
    for i in ivals: fl.noteoff(0, root+i)
    s2 = np.array(fl.get_samples(int(SR*DECAY)), dtype=np.float32)
    fl.delete()

    # scale 16-bit ints → float in [-1.0, 1.0]
    out = np.concatenate([s1, s2]) / 32768.0

    # normalize peak to 0.9 for headroom
    peak = np.max(np.abs(out))
    if peak>0:
        out = out/peak * 0.95

    return out

def main():
    for root in range(MIN_MIDI, MAX_MIDI+1):
        for name, ivals in CHORD_TEMPLATES.items():
            if root + max(ivals) > MAX_MIDI: continue

            tag = f"{note_name(root)}-{name}"
            ogg_file = os.path.join(OUT, tag + ".ogg")

            samples = generate_chord_samples(root, ivals)

            # write scaled & normalized floats into OGG
            sf.write(ogg_file, samples, SR,
                     format="OGG", subtype="VORBIS")

if __name__=="__main__":
    main()
