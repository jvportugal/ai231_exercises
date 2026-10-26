# Long-window Pi benchmark and diagnosis, 26 October 2026

User-reported physical playback run `20261026-192658`, seed 74930, Pi log
`~/vcm_benchmark/vcm_20261026_192834.log`. The user confirmed launching
`--wakeword --term --no-wake-chime --benchmark --vcm-bundle vcm_benchmark_long_window`.
The intended shipped revision is f9d80ea. Actual Pi file hashes and original
capture audio are not available on the development host.

202 wake-word trials: 186 valid commands, 16 OOS; 96 real and 106 synthetic.
An additional 16 no-wake trials are excluded from accuracy. All had no action.

| Metric | Previous structured bundle | Long-window bundle |
| --- | ---: | ---: |
| Intent accuracy | 84.2% (170/202) | 82.7% (167/202) |
| Command accuracy including slot | 81.2% (164/202) | 81.7% (165/202) |
| Real command accuracy | 71.9% (69/96) | 74.0% (71/96) |
| Synthetic command accuracy | 89.6% (95/106) | 88.7% (94/106) |
| OOS false accepts | 10/16 | 8/16 |
| Valid commands rejected | 8/186 | 16/186 |
| Wrong accepted commands | 20/186 | 13/186 |
| Slot exact, conditional on correct intent | 89/95 | 89/91 |
| Extra fires | 0 | 1 |
| Mean inference | 35.9 ms | 46.1 ms |
| Response latency p95 | 0.84 s | 0.83 s |

This is one net additional correct command, not convincing evidence of a
substantial improvement. The conditions differ: speech level -33.4 to -38.1
dBFS, SNR 28.3 to 25.6 dB, and shuffle seed. No paired significance claim.
Current error budget: 37 = 16 false rejections + 13 wrong accepted commands +
8 OOS false accepts. Reaching at least 90% on 202 rows requires 182 correct,
i.e. 17 additional correct rows, with no compensating losses.

Main current confusions: TEMPERATURE to REJECT (6), BRIGHTNESS to REJECT (4),
LIGHT_ON to LIGHT_OFF (2), LIGHT_OFF to VOLUME_DOWN (2), PLAY_MUSIC recall 3/6.
OOS false accepts include juice requests, heating requests, language selection,
flights, washing machine, and a sentence about a sister visiting. Correct-intent
progress markers hide two slot mistakes: trial 69 (18 to 22 degrees) and trial
133 (100 to 20 percent brightness).

The physical report's prose says no synthetic OOS, but its tables report 6.
Use the table counts pending the raw manifest. Its parameter count 3,192,359
and 68.1 MFLOP appear to describe a different model than the 3,110,872-parameter
shipped ensemble. Correct launch confirmed; static metadata is a possibility,
not proof of a wrong model. Original Pi hashes remain needed.

## Frozen-model diagnostics performed locally

No training, calibration, or model substitution was performed in this audit.
Approved dataset HOLDOUT has been examined before; these are diagnostic reuse
results, not fresh independent evaluation. The shipped runtime was run on all
202 source waveforms at fixed gains and at a standardized active-speech RMS.

| Perturbation | Intent accuracy | Real intent accuracy | Changed intent or slot | OOS rejected |
| --- | ---: | ---: | ---: | ---: |
| Original source | 93.07% | 86.46% | 0 | 11/16 |
| Source minus 6 dB | 92.57% | 85.42% | 3 | 11/16 |
| Source minus 12 dB | 92.08% | 84.38% | 2 | 11/16 |
| Each source scaled to -38.1 dBFS active-speech RMS | 90.59% | 82.29% | 9 | 10/16 |

The last condition is a hypothetical level test, not a recreation of the Pi
microphone signal or the benchmark mic-check measurement. Pure gain changes
explain some sensitivity but do not reproduce the 82.7% live intent accuracy.
The runtime detects speech before peak normalization; absolute level therefore
can move speech boundaries despite later normalization.

Numerous source recordings contain 2-4 seconds after the detected active-speech
end. Negative latency measured from file end is compatible with trailing silence;
it does not prove capture truncation. Actual capture waveforms and timestamps
are needed to distinguish file silence, early endpointing, and record matching.

The auxiliary training vocabulary explicitly excluded `on` as a stopword, and
its frequency cutoff did not guarantee command-defining keywords were present.
The main intent loss still supervises LIGHT_ON, so this is a training design
weakness, not proof of the cause of the light errors. New training vocabulary
construction now reserves observed schema/contrast words. Existing checkpoint
vocabularies, weights, and shipped predictions are unchanged.

## Next experiment

Use `--diagnostics-dir ~/vcm_diagnostics` with the confirmed launch command for
a short targeted playback test. Every classified command saves exact float32
NPY audio, listenable WAV, decoder thresholds/confidences/rejection reason,
and session model/runtime SHA256 hashes. Benchmark JSONL remains unchanged.
Replay the same captured waveform through old/new models to separate model
changes from microphone, room, capture, and gain changes. Compare captured and
original versions of approved dataset clips; diagnostic recordings are not
silently added to training.

Prioritize domain-matched paired augmentation and explicit on/off, play/pause,
up/down distinctions. Candidate frontend experiment: PCEN, trained and evaluated
with its matching frontend. Candidate rejection experiment: independent support
for the predicted command, tested against held-out speakers and unsupported
speech. These are proposals, not demonstrated improvements. Previous
contrastive/evidence-attention experiments failed joint acceptance criteria;
merely adding those losses again is not a new solution. Any added ensemble
member must demonstrate complementary errors and respect the parameter cap.
