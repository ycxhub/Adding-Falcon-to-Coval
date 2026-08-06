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

## TTS WER

`falcon2_tts_wer_{raw,summary}_full_fp32.csv` and the `trimmed_fp32` pair score
the same WAVs on Coval’s TTS WER metric: transcribe the synthesized audio with a
fixed ASR model, then score that transcript against the synthesis input under
`EnglishTextNormalizer` + `jiwer` (`norm_version` 2). Produced by
`tools/tts_wer.py` — see `tools/README.md` for the command.

- All 9 Falcon 2 voices and the Gen2 control transcribe the prompt exactly:
  **0.00 % WER**, on both the full and lead-in-trimmed audio.
- Only 10 reference words per voice (one prompt), so this bounds WER rather
  than resolving differences between voices. The 30-prompt `tts-v1` set needs
  `MURF_API_KEY` to synthesize.
- Scored with local Whisper large-v2 at **float32**, a stand-in for Coval’s
  `whisper-1`. At `int8` the same clips return 2–15 % WER from quantization
  hallucinations (Korean/German tokens, repetition loops) rather than real TTS
  errors — reproduce with `--compute-type int8` to see it. Anyone re-running
  this offline should stay at float32.
