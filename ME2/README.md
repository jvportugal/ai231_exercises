# ME2 — RV90 End-to-End Voice Command Model

ME2 is a local, hands-free voice-control system built around the **RV90**
voice-command model and designed primarily for a Raspberry Pi 5. Say
**“Destiny”**, wait for the acknowledgement chime, speak a command, and the
system classifies and executes that command.

The complete runtime is included in this folder:

- a Destiny wake-word ONNX model;
- the 19-intent, two-head RV90 ONNX model and its decoding policy;
- the exact mel filterbanks and label maps required by both models;
- persistent microphone capture and silence endpointing;
- a generated two-note acknowledgement chime;
- graphical and terminal action interfaces;
- Raspberry Pi GPIO control for a common-cathode RGB LED;
- bundled music samples;
- live Quezon City weather with an offline simulation fallback; and
- diagnostic programs and deterministic tests.

Speech and inference run locally. No cloud speech-recognition service is used.
Internet access is needed during setup to install standard dependencies and
while retrieving live weather, but not for wake-word or command inference.

## 1. System flow

~~~text
microphone
   │
   ▼
adaptive speech gate ── filters steady room/fan noise
   │
   ▼
Destiny wake-word model
   │
   ├── no match ──► continue listening
   │
   └── match ─────► play acknowledgement chime
                         │
                         ▼
                 capture one command
                 until trailing silence
                         │
                         ▼
                 RV90 intent + slot model
                         │
                         ▼
                 action dispatcher
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
        GPIO          local UI       weather API
       /audio         /timers        or simulation
~~~

The wake-word and RV90 models are separate. The first decides whether the user
said “Destiny.” RV90 predicts one of 19 actions and, where required, one of 18
slot values such as a timer duration, color, or temperature.

## 2. Intended platform

### Primary target

- Raspberry Pi 5
- 64-bit Raspberry Pi OS with the APT package manager
- Python 3
- USB or compatible ALSA microphone
- speaker, headphones, HDMI audio, or another PortAudio output
- optional desktop display for the graphical Game Boy-inspired interface
- optional four-pin common-cathode RGB LED

The ONNX models run on CPU; a GPU is not required.

### Development and demonstration platforms

The core inference and action application is intended to run on Linux, macOS,
and Windows. GPIO automatically falls back to a clearly reported simulation
when Raspberry Pi hardware is not available. Use terminal mode on machines
without a graphical display. The Space-key controls in the two standalone
terminal testers are Unix-oriented; on Windows, Enter may be required.

The provided setup script directly supports Raspberry Pi OS/Debian-family
systems and macOS with Homebrew. Windows users must install Python, PortAudio,
Tk, and the packages in **requirements.txt** manually.

## 3. Hardware

The minimum hardware for voice operation is a computer, microphone, and audio
output. The RGB LED is optional.

### RGB LED wiring

The physical light action uses these BCM pins:

| LED channel | Raspberry Pi BCM pin |
|---|---:|
| Red | 17 |
| Green | 27 |
| Blue | 22 |
| Common cathode | Ground |

Place an appropriate current-limiting resistor, commonly 220–330 Ω, in series
with each color channel. Do not connect LED color channels directly to GPIO
without resistors. The supplied driver expects a **common-cathode** LED and
uses active-high outputs.

## 4. Fresh installation

These instructions reproduce the runnable system from a new checkout.

### Clone and enter ME2

~~~bash
git clone https://github.com/jvportugal/ai231_exercises.git
cd ai231_exercises/ME2
~~~

### Run the setup script

~~~bash
chmod +x setup.sh
./setup.sh
source .venv/bin/activate
~~~

Alternatively, the all-in-one launcher makes `setup.sh` executable, runs the
complete setup, and starts the Destiny wake-word application. From **ME2**:

~~~bash
bash start.sh
~~~

After its first run, it can also be invoked as `./start.sh`. Any extra options
are passed to the action runner; for example, `./start.sh --term` deliberately
uses the terminal interface.

On Raspberry Pi OS, setup uses APT and may ask for the sudo password. It
installs Python/Tk, PortAudio, ALSA tools, compilation prerequisites, and the
Pi GPIO backend. Python packages are isolated inside **ME2/.venv**.

Setup also:

1. verifies that sounddevice/PortAudio can enumerate audio devices;
2. checks Tk;
3. compiles the main Python programs;
4. verifies every required model asset; and
5. loads and warms up both bundled ONNX models.

Rerunning setup is safe. The virtual environment and downloaded dependencies
are intentionally not committed to Git.

### Verify the bundled assets

From the ME2 directory:

~~~bash
sha256sum -c checksums.sha256
~~~

The checksum manifest covers both models, both filterbanks, model metadata,
RV90 labels and decoding policy, and the two bundled music files.

## 5. Run the complete system

After activating the virtual environment:

~~~bash
python3 vcm_action_runner/runner.py --wakeword
~~~

The normal interaction is:

~~~text
WAITING FOR DESTINY
→ user says “Destiny”
→ DESTINY DETECTED
→ two-note acknowledgement chime
→ LISTENING FOR COMMAND
→ user speaks one command
→ trailing silence ends recording
→ PROCESSING COMMAND
→ intent and optional slot are predicted
→ action runs
→ RETURNING TO WAKE-WORD MODE
~~~

Speak the command after the chime. The microphone remains open, but blocks
that overlap the chime are discarded so the cue is not classified as part of
the command.

### Display selection

The graphical interface is selected automatically when a display is present.
It uses a four-shade, Game Boy-inspired LCD design.

~~~bash
python3 vcm_action_runner/runner.py --wakeword --gui
python3 vcm_action_runner/runner.py --wakeword --term
~~~

Use terminal mode over SSH without X forwarding. To show the GUI remotely,
the laptop must have an X server running and must request X11 forwarding when
it opens the connection. The setup script installs `xauth` and enables the
Pi-side SSH setting, but it cannot add forwarding to an SSH connection that is
already open.

From the laptop, connect and verify the forwarded display:

~~~bash
ssh -Y USER@rpi5.local
echo "$DISPLAY"                 # normally localhost:10.0 (or similar)
cd ~/ai231_exercises/ME2
./start.sh
~~~

Linux desktops normally already have an X server. macOS needs XQuartz running.
On Windows, use an SSH environment with an X server, such as MobaXterm, VcXsrv,
or WSLg. X11 forwarding carries the window only: the chime and music still use
the Raspberry Pi's selected HDMI, USB, Bluetooth, or other audio output.

### Other operating modes

Keyboard-controlled recording, useful for command-model testing:

~~~bash
python3 vcm_action_runner/runner.py --voice
~~~

Typed intent codes, useful when testing actions without a microphone:

~~~bash
python3 vcm_action_runner/runner.py
~~~

Press Ctrl+C to stop wake-word mode. In typed mode, enter q to quit.

## 6. Supported actions

The VCM has 20 intent-head classes: 19 actions plus REJECT. Six actions use the
second model head to select one of three supported values.

| Code | Action | Supported values or example |
|---|---|---|
| P | Play music | “Play music” |
| W | Weather | “What’s the weather?” |
| T | Time | “What time is it?” |
| ON | Light on | “Turn on the lights” |
| OFF | Light off | “Turn off the lights” |
| PA | Pause | “Pause music” |
| S | Stop | “Stop music” |
| N | Next | “Next song” |
| VU | Volume up | “Increase the volume” |
| VD | Volume down | “Decrease the volume” |
| C | Call | “Make a call” |
| M | Message | “Send a message” |
| LR | List reminders | “Show reminders” |
| TM | Timer | 10 seconds, 30 seconds, 1 minute |
| AL | Alarm | 6:00 AM, 8:00 AM, 9:00 PM |
| TE | Temperature | 18, 22, 26 degrees |
| BR | Brightness | 20, 60, 100 percent |
| CO | Color | Red, Blue, Green |
| CR | Create reminder | Drink water, Study, Exercise |

The training schema contains 93 command phrasings. The table shows canonical
examples and the allowed slot values, not every wording in the dataset.
Out-of-scope or sufficiently uncertain audio is assigned REJECT and does not
trigger an action.

## 7. Action behavior

The action layer mixes real local operations with demonstrations:

- music playback, the countdown timer, the clock, and live weather are
  functional local features;
- light, color, and brightness control real GPIO only when compatible
  Raspberry Pi hardware is available, and otherwise report simulation mode;
- temperature, alarms, calls, and messages are interface simulations only;
  they do not control a thermostat, schedule an operating-system alarm, place
  a phone call, or send a message; and
- reminders are stored only in memory for the current process and are not
  persisted after the application exits.

### Light

On a Raspberry Pi, **actions/light_driver.py** controls BCM 17/27/22 through
gpiozero and lgpio. On other machines—or when GPIO initialization fails—it
prints SIMULATED mode and leaves the rest of the pipeline operational.

Run the diagnostic if the LED does not respond:

~~~bash
python3 vcm_action_runner/diagnose_light.py
~~~

### Music and wake chime

The two demonstration tracks are included under
**vcm_action_runner/assets/music/**. The default wake chime is generated
mathematically by **wakeword.py**, so it requires no downloaded audio file.

~~~bash
python3 vcm_action_runner/runner.py --wakeword --wake-chime-volume 0.15
python3 vcm_action_runner/runner.py --wakeword --no-wake-chime
python3 vcm_action_runner/runner.py --wakeword --wake-chime path/to/custom.wav
~~~

A custom cue must be an uncompressed 8-, 16-, or 32-bit PCM WAV. The generated
cue remains the self-contained default.

### Weather

The WEATHER action requests current conditions for fixed Quezon City
coordinates from the Open-Meteo forecast endpoint. No API key is required.
The panel displays:

- **OBSERVED** — the timestamp associated with the provider’s current data;
- **RETRIEVED** — when this device completed the request;
- temperature and apparent temperature;
- humidity, cloud cover, precipitation, and wind; and
- the source and live/simulated status.

Times are displayed in Philippine Time. Successful responses are marked
**LIVE // N-MIN MODEL**, where N is the interval reported by the provider
(normally 15). A timeout, network error, provider failure, or invalid response
automatically produces an explicitly marked
**SIMULATED // OFFLINE** panel. Simulation is never presented as live data.

Only the fixed coordinates and normal HTTP request metadata leave the device.
All speech remains local.

### Domain hand-off

Actions are grouped into LIGHT, MUSIC, and OTHERS. Moving between groups stops
the previous group’s active hardware:

- leaving LIGHT turns the LED off but preserves color and brightness;
- leaving MUSIC stops playback but preserves track and volume; and
- leaving OTHERS closes the active panel.

This prevents music or a light from continuing accidentally after an unrelated
command.

## 8. Wake-word testing and tuning

Test Destiny without executing actions:

~~~bash
python3 wakeword_tester.py
~~~

Press Space once to start and again to stop. Test:

- monotone, first-syllable stress, and second-syllable stress;
- close, normal, and farther microphone distances;
- quiet and loud voices;
- unrelated words and normal conversation; and
- the actual room with fans or air conditioning running.

The runtime default threshold is 0.850. Raise it to reduce false activations,
at the cost of more missed wake words:

~~~bash
python3 wakeword_tester.py --threshold 0.90
python3 vcm_action_runner/runner.py --wakeword --wakeword-threshold 0.90
~~~

The detector estimates the room floor at startup. If distant speech does not
start a candidate, reduce the absolute gate slightly:

~~~bash
python3 vcm_action_runner/runner.py --wakeword --speech-rms 0.006
~~~

If changing fan noise starts too many candidates, increase the adaptive
multiplier:

~~~bash
python3 vcm_action_runner/runner.py --wakeword --noise-floor-multiplier 1.9
~~~

Change one setting at a time and use the standalone tester before comparing
results.

## 9. Tests

### Setup/model smoke test

The final section of **setup.sh** loads and warms both ONNX models.

### Weather tests

~~~bash
python3 -m unittest -v vcm_action_runner.tests.test_weather
~~~

These tests use a fixed provider response and do not require internet. They
cover live parsing, Philippine timestamps, response validation, network
failure handling, and simulation labeling.

### Action-runner self-test

~~~bash
python3 vcm_action_runner/selftest.py
~~~

This drives all 19 actions through the real dispatcher in terminal mode and
checks shared device state and cross-domain hand-off. It can take roughly a
minute because action transitions deliberately wait for cancellation.

### Command-model microphone test

~~~bash
python3 tester.py
~~~

This tests microphone capture and VCM classification without running actions.

### macOS check

~~~bash
python3 mac_check.py
~~~

This checks the same command model and feature pipeline used on the Pi, either
from the live microphone or from a WAV supplied with **--audio**. It does not
test the GUI. The graphical application separately keeps its single Tk
instance on the main thread for macOS compatibility.

## 10. Troubleshooting

### No microphone is listed

~~~bash
arecord -l
python3 -c "import sounddevice as sd; print(sd.query_devices())"
~~~

Confirm the correct input appears, then pass its sounddevice number using
**--mic-device NUMBER**.

### The chime or music is silent

Check the selected system output and test ALSA:

~~~bash
speaker-test -t sine -f 880 -l 1
~~~

On desktop systems, confirm that Python/Terminal has microphone permission and
that the intended speakers are the default output.

### The GUI does not appear over SSH

Use **--term**, connect with SSH X forwarding, or launch from the Pi desktop.
The VCM continues to work in terminal mode.

### The light reports SIMULATED mode

Rerun setup, log out and back in after the gpio group change, then run
**diagnose_light.py**. Also verify common-cathode wiring and BCM pin numbers.

### Weather always shows SIMULATED // OFFLINE

Confirm the Pi can reach HTTPS sites and that its clock is correct. Weather
failures do not affect wake-word, VCM, or other actions.

### Destiny activates on noise

Use **wakeword_tester.py**, try a threshold of 0.90 or 0.95, and slightly raise
the noise-floor multiplier. Do not tune only with positive examples; leave the
tester running during ordinary room noise and unrelated speech.

## 11. Project layout

| Path | Purpose |
|---|---|
| **setup.sh** | Installs and verifies the complete runtime |
| **requirements.txt** | Cross-platform Python runtime dependencies |
| **wakeword_tester.py** | Standalone Destiny detector evaluation |
| **tester.py** | Standalone command-VCM microphone test |
| **vcm_infer.py** | VCM preprocessing and ONNX inference |
| **wakeword_model/** | Destiny ONNX model, filterbank, and metadata |
| **vcm_model/** | RV90 ONNX model, filterbank, labels, and decoding policy |
| **training/** | Training, evaluation, export source, exact splits, reports, and retained checkpoints for both models |
| **vcm_action_runner/runner.py** | End-to-end dispatcher and CLI |
| **vcm_action_runner/wakeword.py** | Persistent mic, chime, wake scoring, endpointing |
| **vcm_action_runner/voice.py** | Command capture and VCM bridge |
| **vcm_action_runner/actions/** | Weather, audio, lighting, timers, and simulations |
| **vcm_action_runner/ui/** | Game Boy-style Tk UI and terminal panels |
| **vcm_action_runner/assets/** | Bundled local media |
| **checksums.sha256** | Integrity values for model/media assets |

## 12. Reproducibility scope

This repository reproduces the complete **inference and action runtime**. A
fresh user can clone the repository, run setup, and execute the same exported
models and RV90 policy without locating replacement models, labels,
filterbanks, songs, or a wake cue.

The training, evaluation, and export programs, exact wake-word split manifests,
local noise inputs, retained PyTorch checkpoints, and recorded reports are
committed under **training/**. This supports both checkpoint-to-ONNX export and
fresh training after the source corpora are obtained. The virtual environment,
operating-system packages, and multi-gigabyte third-party corpora are not
committed. Setup downloads standard runtime dependencies appropriate to the
target machine. Source dataset licenses and consent conditions must be followed
separately when obtaining or retraining on the corpora.

## 13. How the models were created

This section is intentionally last so installation and operation come first.

### 13.1 Destiny wake-word dataset

The wake model was trained without requiring recordings from the end user. Its
source was the English portion of the
[MLCommons Multilingual Spoken Words Corpus](https://mlcommons.org/datasets/multilingual-spoken-words/),
which consists of keyword clips extracted from crowd-sourced speech and is
released under CC BY 4.0.

The curated, speaker-disjoint split used:

| Split | Positive Destiny clips | Positive speakers | Negative clips | Negative speakers |
|---|---:|---:|---:|---:|
| Training | 1,010 | 668 | 3,663 | 413 |
| Held out | 449 | 286 | 3,234 | 254 |

Negatives included other commands, phonetically near words, and random words.
Held-out speakers were never sampled during training.

#### Wake-word preprocessing

1. Convert to 16 kHz mono and remove DC mean.
2. Energy-VAD crop the word while retaining 65 ms edge padding.
3. Randomly place or center it in a one-second, 16,000-sample window.
4. Calculate a 40-bin mel spectrogram with a 512-point FFT, 160-sample hop,
   35–7,600 Hz range, and 100 time frames.
5. Convert power to dB with an 80 dB range.
6. Apply utterance-level mean/variance normalization.
7. Add a second temporal-delta channel.

#### Wake-word alterations and augmentation

Original corpus files were not modified. Variants were created in memory:

- whole-word pitch shift from −3.5 to +3.5 semitones;
- whole-word rate from 0.78× to 1.24×;
- independently warped first/second word halves from 0.72× to 1.28× to vary
  syllable timing and stress;
- gain from −22 dB to +16 dB;
- randomized placement inside the model window;
- synthetic room impulse responses for far-field speech;
- close-microphone bass/proximity coloration;
- short plosive bursts, soft saturation, overload, and hard clipping;
- colored background noise; and
- procedural fan/air-conditioner hum containing 50/60 Hz mains components,
  blade harmonics, colored airflow, and slow modulation, mixed at 2–32 dB SNR.

One third of generated negative examples were fan-only hard negatives. Other
negative words received similar channel transformations so augmentation
artifacts could not become a positive-class shortcut.

#### Wake-word model and optimization

RobustWakeCNN-v2 has 73,410 parameters. It contains three 2-D convolution
blocks, an order-aware temporal convolution stack, attention and max temporal
pooling, and a small binary classifier.

Training used:

- random seed 231023;
- balanced positive/negative online sampling;
- 4,200 generated examples per epoch;
- batch size 96;
- AdamW, learning rate 1.2e-3, weight decay 2e-4;
- cosine learning-rate decay;
- binary cross entropy with logits;
- gradient-norm clipping at 5; and
- at most 18 epochs with patience 6.

Model selection combined clean recall and deterministic stress recall while
penalizing fan false positives. The held-out stress set covered far speech with
fan noise, close clipped speech, high/low pitch and prosody, room reverberation,
and 600 fan-only examples.

At the deployed 0.850 threshold, the recorded evaluation was 94.65% clean
Destiny recall, 91.75% stress recall, 1.24% held-out out-of-scope
false-positive rate, and 0/600 procedural fan false activations. These are
dataset measurements, not guarantees for every microphone and room.

The selected PyTorch checkpoint was exported to ONNX opset 17. The matching
mel filterbank and evaluation metadata are included in **wakeword_model/**.

### 13.2 RV90 voice-command dataset

RV90 was created from
[AI231 ME2 Voice Commands](https://huggingface.co/datasets/airimonda/ai231-me2-voice-commands)
(DOI 10.57967/hf/10723). The corpus defines 93 phrasings for the 19-intent
Option B command schema plus out-of-scope audio. Files are standardized as
16 kHz, mono, 16-bit PCM WAV, and real speakers and synthetic voice identities
are separated across the train, test, and holdout splits.

The published base corpus reports:

| Split | Clips | Duration | Dataset role |
|---|---:|---:|---|
| Train | 10,682 | 6.4 h | Command and out-of-scope training audio |
| Test | 4,418 | 2.5 h | Speaker-disjoint benchmark audio |
| Holdout | 196 | 0.2 h | Raspberry Pi/live benchmark audio |
| Numerals | 66,390 | 21.8 h | Separate number-only pool |

The corpus combines course recordings, synthetic voices, SLURP, Google Speech
Commands v2, Mozilla Common Voice 19 English, Fluent Speech Commands, SNIPS,
Timers and Such, MLEnd spoken numerals, the Multi-Sensor Voice Command dataset,
DEMAND/MS-SNSD noise, and Filipino-accented supplemental synthetic speech.

The compilation has mixed source licenses and is limited as a whole to
research and education/non-commercial use. Group recordings and some
Filipino-accented synthetic references have additional consent or provenance
conditions. Consult the dataset card and each source license before obtaining,
redistributing, or retraining on the corpus.

#### Dataset preparation

The following preparation produced the data consumed by RV90:

1. Standardize every waveform to 16 kHz mono PCM and normalize the schema to
   19 command intents, 18 valid slot tokens, and REJECT.
2. Split by real speaker and synthetic voice identity before balancing so an
   identity cannot cross the train, test, and holdout partitions.
3. Transcribe real course recordings with Whisper large-v3-turbo and compare
   the transcript with the assigned label. Move wrong commands, background
   speech, silence, and unsupported slot values to out of scope.
4. Remove unslotted commands that include unsupported slot information and
   replace shortages only with voices already assigned to that split.
5. Balance the 93 phrasings with real audio first and same-split synthetic
   voices where additional coverage is needed.
6. Build out-of-scope coverage from noise, Filipino speech, near-miss requests,
   and general speech.
7. Generate hard negatives from noise-only audio, three-to-six-speaker babble,
   reversed clips, the first 25–40% of incomplete commands, and near-silence
   hiss.
8. Add 14,120 Filipino-accented synthetic command clips to training only.
   Tune-out-of-scope identities remain outside the fitting data.

RV90 fitting used 25,941 rows: 10,733 base training rows, 14,120
Filipino-accented supplemental rows, 1,000 generated hard negatives, and 88
noise rows. The development/selection pool contained 4,443 test rows and 400
speaker-disjoint tune-out-of-scope examples. These are observed counts from
the derived RV90 working parquets, not the smaller published-base counts in
the table above. Although the dataset card assigns the 88 noise files an
augmentation-only role in its canonical recipe, this RV90 recipe explicitly
repurposed them as out-of-scope fitting rows.

#### Feature pipeline and augmentation

At training and inference time, the feature path is:

1. Decode or resample to 16 kHz mono.
2. Center a 2.5-second, 40,000-sample window on the speech region.
3. Compute a 40-bin power mel spectrogram with a 400-sample FFT and Hann
   window, 160-sample hop, and 20–8,000 Hz frequency range.
4. Apply base-10 log scaling, map the fixed −10.0 to 1.4 range into 0–1, and
   clip.
5. Apply per-utterance, per-mel mean/variance normalization, rescale with
   `(x + 3) / 6`, and clip again.
6. Supply a `1 × 40 × 250` float32 tensor to the network.

The primary fitting run used online waveform augmentation:

- a 50% chance of DEMAND/MS-SNSD room noise at 12–28 dB SNR;
- random gain within ±4 dB;
- a time shift of up to ±10% of the clip; and
- a 30% chance of procedural three-to-five-source babble at 8–20 dB SNR.

Sampling and loss weighting increased the contribution of real recordings,
the six acoustically similar command pairs (`VOLUME_UP`, `VOLUME_DOWN`,
`LIGHT_ON`, `LIGHT_OFF`, `STOP`, and `NEXT`), rare classes, and REJECT.

The refinement stage used the same 25,941 rows with source-balanced sampling:
30% course real voice, 20% other real speech, 35% synthetic commands, and 15%
REJECT. It added mild babble with probability 0.10 and 0.94–1.06× speed
perturbation with probability 0.10. Knowledge distillation against the primary
checkpoint supplied 70% of the refinement objective. The deployed parameter
state is an 80/20 interpolation of the primary and epoch-three refinement
states; it remains one ordinary network with no ensemble at inference time.

#### RV90 architecture

RV90 is a 3,192,807-parameter convolutional recurrent neural network:

~~~text
input: B × 1 × 40 × 250 log-mel features
  │
  ├─ Conv2d 1→32,  3×3, stride 3, padding 1 → BatchNorm → ReLU
  ├─ Conv2d 32→64, 3×3, stride 3, padding 1 → BatchNorm → ReLU
  └─ Conv2d 64→128, 3×3, stride 3, padding 1 → BatchNorm → ReLU
       │
       └─ mean over the remaining mel-frequency axis
            │
            └─ 3-layer bidirectional GRU, 256 units per direction
                 │
                 └─ final time step → Linear 512→256 → ReLU → Dropout 0.3
                      ├─ intent head: 20 logits (19 actions + REJECT)
                      └─ slot head:   19 logits (18 values + REJECT)
~~~

Primary optimization used seed 301, batch size 128, AdamW at `3e-4` with
weight decay `1e-4`, cosine learning-rate decay, weighted intent cross entropy,
slot cross entropy, and early stopping with patience six. The refinement used
seed 812, four 10,000-sample epochs, batch size 64, AdamW at `2e-5`, a 0.1×
learning-rate multiplier for convolution weights, frozen BatchNorm running
statistics, cosine decay, and gradient-norm clipping at 5. Neural-weight and
checkpoint selection used the dataset's test split as validation data together
with the separate tune-OOS pool. Holdout audio was used only for evaluation. It
was not used for training or fine-tuning, checkpoint selection, interpolation,
rejection-policy configuration, or model selection.

#### RV90 decoding and rejection

The two output heads are decoded jointly. For each command, the decoder adds
`0.075 ×` the log-probability mass of slot values compatible with that command
to its intent logit. It then selects a slot only from the values permitted for
the chosen intent. Finally, the REJECT logit is compared with the selected
command using the class-specific margins in **vcm_model/policy.json**. This
policy adds no learned parameters and requires no second model pass.

`vcm_infer.py` applies this policy automatically. Raw independent argmax of the
two ONNX outputs does not reproduce RV90 behavior.

#### Recorded evaluation

| Benchmark | Intent accuracy | End-to-end accuracy or rejection |
|---|---:|---:|
| 4,443-row test/selection set, all voices | 92.010% | 91.222% |
| Test/selection set, 824 real clips | — | 65.655% (541/824) |
| Test/selection set, 3,619 synthetic clips | — | 97.043% |
| 202-row holdout evaluation set, all voices | 94.554% | 94.059% |
| Holdout evaluation set, 96 real clips | — | 90.625% (87/96) |
| Holdout evaluation set, 106 synthetic clips | — | 97.170% |
| 250-row out-of-scope evaluation benchmark | — | 87.200% rejected (218/250) |

These are recorded measurements from the retained verification run. The
dataset's test split and tune-OOS pool were used for neural-model validation
and selection. The holdout was evaluated only after the neural weights,
interpolation, and rejection policy were fixed; it did not inform training,
fine-tuning, checkpoint selection, policy configuration, or model selection.
The deployed inference assets are checksum-verifiable here; the training and
evaluation pipelines are not included in this repository.

The ONNX export matches the PyTorch checkpoint: maximum absolute error was
`3.81e-06` for the intent head and `4.77e-06` for the slot head, with matching
argmax outputs. A single-thread x86 benchmark measured 1.996 ms median ONNX
inference plus 0.142 ms median policy decoding for batch size one. Those host
figures are relative measurements, not Raspberry Pi latency claims.

RV90 is exported as an ONNX opset-13 graph with a dynamic batch axis. The
deployed directory contains only the current model and its required artifacts:

- **vcm_model/vcm.onnx** — RV90 network;
- **vcm_model/melbank.npy** — exact 40-bin preprocessing filterbank;
- **vcm_model/labels.json** — intent and slot-token order; and
- **vcm_model/policy.json** — constrained decoder and REJECT margins.

Audio preprocessing and policy decoding stay in **vcm_infer.py**, so the
Raspberry Pi runtime uses NumPy and ONNX Runtime rather than the PyTorch
training stack.
