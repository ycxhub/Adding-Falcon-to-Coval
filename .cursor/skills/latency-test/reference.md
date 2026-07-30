# Coval TTFA reference

Source of truth: [coval-ai/benchmarks](https://github.com/coval-ai/benchmarks)
methodology (`docs/methodology.md`) and
`runner/src/coval_bench/metrics/ttfa.py`.

## Perceived TTFA definition

```
TTFA = (first audio chunk arrival − synthesis start)
     + leading silence inside the stream before the first audible sample
```

Goal: measure what an enqueue-and-play client waits before hearing sound.

| Term | Meaning |
| --- | --- |
| Arrival | Wall-clock (`time.monotonic`) from synthesis trigger to first non-empty audio chunk |
| Leading silence | Offset to first audible sample via RMS onset detector on assembled PCM |
| Degradation | If offset fails, TTFA = arrival only; keep the audio |

## Onset detector (do not change)

From Coval `ttfa.py`:

| Constant | Value |
| --- | --- |
| `_AUDIBLE_RMS_THRESHOLD` | `0.01` (normalized float PCM in [-1, 1]) |
| `_FRAME_SECONDS` | `0.010` (10 ms) |
| `_HOP_SECONDS` | `0.001` (1 ms) |
| PCM format | Signed 16-bit LE mono; divide by `32768.0` |
| DC handling | Per-frame mean removed before RMS; edge-pad (not zero-pad) |

## Pre-t0 exclusion (latency measurement convention)

TTFA measures model inference / generation latency, **not** connection setup.

| Cohort | Excluded from t0 |
| --- | --- |
| WebSocket streaming | TLS + WS upgrade + optional session/voice setup RTT |
| HTTP TTS | TLS + TCP via a pre-warmed shared client (HTTP/2 preferred) |

Start the clock only when sending the text / starting generation.

## Aggregation

Coval scheduled runs sample `dataset_sample_size` items (default **10**) from
`tts-v1` and share that sample across models within a run. For this skill:

- Run **≥ 10** successful trials
- Report **median, mean, min, max** of perceived TTFA
- Prefer median as the headline (latency distributions are skewed)

## Default prompts (`tts-v1`)

Sample 10 of these (or cycle all 30 across longer runs). IDs match Coval.

1. A1 — Your order #ORD-24589 shipped via UPS tracking 1Z999AA1234567890 and will arrive between 2:30 PM and 4:45 PM tomorrow.
2. A2 — There's a slight delay with your $347.89 order, but we expect it to ship by Friday afternoon.
3. A3 — Order status update: your items are now in "Quality Check" and should ship within 24-48 hours.
4. A4 — Hi Ms. Garcia, your appointment with Dr. Peterson is scheduled for Tuesday, March 5th at 10:30 AM.
5. A5 — I can reschedule you to either 11:15 AM on Wednesday or, let me check, 2:45 PM on Thursday.
6. A6 — Please arrive 10 minutes early and bring your insurance card and a photo ID.
7. A7 — Let's fix your Wi-Fi: unplug your router for exactly 30 seconds, then wait for the green light.
8. A8 — Your device is downloading update version 12.4.1, about 250 MB remaining, roughly 8 minutes left.
9. A9 — I'm seeing error code E-1047 here, which usually means, actually, let me guide you through the solution.
10. A10 — For security, please confirm the last 4 digits of the card ending in 8429 used for your recent purchase.
11. A11 — I need to verify your account: what's the ZIP code we have on file? Please take your time.
12. A12 — We detected 2 unusual login attempts from Chicago, Illinois yesterday around 11:30 PM.
13. A13 — Hi David, your premium subscription renews in 5 days at $29.99, would you like to continue or explore other options?
14. A14 — Based on your purchase history, you might like our flash sale: 30% off electronics through Sunday.
15. A15 — I noticed you haven't logged in since January 15th, is there anything we can help you with?
16. A16 — Your invoice shows a balance of $187.50, including the monthly fee plus tax.
17. A17 — The $24.99 charge is for the premium support package you activated on January 22nd.
18. A18 — Your auto-payment of $67.50 failed due to an expired card, would you like to update your payment method?
19. A19 — Thanks for your interest in our software, companies your size typically see 20-25% efficiency gains.
20. A20 — For a 50-person team, I'd recommend our Business plan at $149 per month rather than Basic.
21. A21 — Great timing! We're offering 20% off annual plans, that saves you $359.88 for the first year.
22. A22 — On a scale of 1 to 10, how would you rate your current pain level?
23. A23 — Have you taken any medications today? Please include prescription drugs and over-the-counter items.
24. A24 — Your symptoms sound manageable, but I'd recommend seeing Dr. Martinez within 48 hours to be safe.
25. A25 — Your PTO request for December 20th through 24th has been approved, that's 3 business days.
26. A26 — Password reset complete: your temporary password was sent to your email. Please change it within 24 hours.
27. A27 — IT ticket about your printer issue has been assigned to technician Lisa Chen.
28. A28 — Your wire transfer of $1,247.50 to account ending in 5691 was processed at 2:15 PM.
29. A29 — Hi Mr. O'Connor, I'm calling about your consultation, are you available for 20 minutes between 9 AM and noon?
30. A30 — System alert: your server usage hit 87% at 3:42 AM, but it's back to normal levels now.

Fixed-prompt alternative (voice A/B, silence forensics):

`Your order shipped this morning and should arrive by Friday.`

## Caveats

- Research methodology: absolute ms may not match production UX or the public leaderboard without Coval’s full runner + pinned commit.
- Numbers before Coval’s 2026-05 perceived-TTFA change are not comparable (arrival-only vs arrival + silence).
- Provider-side variance is expected; median across ≥10 trials is more informative than a single shot.
