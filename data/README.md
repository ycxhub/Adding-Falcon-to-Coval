# Data — Falcon 2 leading-silence measurements

Artifacts from validating Murf Falcon 2 against Coval’s perceived TTFA metric
(`arrival + leading silence` before first audible sample at 0.01 RMS).

## Layout

```
data/
  images/          Charts from the 100-call voice matrix and lead-in forensics
  csv/             Raw + per-voice summary for the 10×10 matrix
  audio/
    full/          One synthesis per voice (PCM WAV, 24 kHz mono)
    trimmed/       Same files with leading silence removed (Coval onset)
    leadin_amplified/  Gain-boosted lead-in only (Natalie, Miles) for listening
    sample_metadata.csv  Onset ms and byte sizes for full/trimmed samples
```

## Voices

Natalie, Miles, Gordon, Amara, Madison, Tyler, Will, Grant, Judith (Falcon 2),
and Ken (Gen2 control).

## Text used

`Your order shipped this morning and should arrive by Friday.`

## Notes

- Matrix CSVs: 10 rounds × 10 voices = 100 measured calls, 0 failures.
- `audio/full` and `audio/trimmed` are a separate one-shot capture per voice
  (not the matrix runs), so onset ms may differ slightly from CSV means.
- Trim uses Coval’s detector: 10 ms RMS window, 1 ms hop, threshold 0.01.
