# ME2 — End-to-End Voice Command Model

ME2 is a local, hands-free voice-control system designed primarily for a
Raspberry Pi 5. Say **“Destiny”**, wait for the acknowledgement chime, speak a
command, and the system classifies and executes that command.

The complete runtime is included in this folder:

- a Destiny wake-word ONNX model;
- a 19-intent, two-head voice-command ONNX model;
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
                 VCM intent + slot model
                         │
                         ▼
                 action dispatcher
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
        GPIO          local UI       weather API
       /audio         /timers        or simulation
~~~

The wake-word and VCM models are separate. The first decides whether the user
said “Destiny.” The second predicts one of 19 actions and, where required, one
of 18 slot values such as a timer duration, color, or temperature.

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

The software can also run on Linux, macOS, and Windows. GPIO automatically
falls back to a clearly reported simulation when Raspberry Pi hardware is not
available. Use terminal mode on machines without a graphical display.

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

On Raspberry Pi OS, setup uses APT and may ask for the sudo password. It
installs Python/Tk, PortAudio, ALSA tools, compilation prerequisites, and the
Pi GPIO backend. Python packages are isolated inside **ME2/.venv**.

Setup also:

1. verifies microphone/PortAudio access;
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
labels, and the two bundled music files.

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
connect with X forwarding or run the application directly on the Pi desktop.

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
**LIVE // 15-MIN MODEL**. A timeout, network error, provider failure, or
invalid response automatically produces an explicitly marked
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

The graphical application uses one main-thread Tk instance, avoiding the macOS
failure caused by constructing windows on worker threads.

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
| **vcm_model/** | Command ONNX model, filterbank, and labels |
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
models without locating replacement models, labels, filterbanks, songs, or a
wake cue.

The virtual environment, operating-system packages, and multi-gigabyte source
training corpora are not committed. Setup downloads standard runtime
dependencies appropriate to the target machine. The following model-creation
section records the data sources and exact shipped recipes; source dataset
licenses must be followed separately if retraining.

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

### 13.2 Voice-command dataset

The command model used
[AI231 ME2 Voice Commands](https://huggingface.co/datasets/airimonda/ai231-me2-voice-commands)
(DOI 10.57967/hf/10723). Its release contains 93 command variations for the
19-intent Option B schema and an out-of-scope class. Audio was standardized as
16 kHz, mono, 16-bit PCM WAV.

The base release reports:

| Split | Clips | Duration | Purpose |
|---|---:|---:|---|
| Train | 10,682 | 6.4 h | Model fitting |
| Test | 4,418 | 2.5 h | Speaker-disjoint evaluation |
| Holdout | 196 | 0.2 h | Final Raspberry Pi/live evaluation |
| Numerals | 66,390 | 21.8 h | Separate number-only pool |

No real speaker or synthetic voice appears in more than one train/test/holdout
split. The dataset combines course recordings, synthetic voices, and these
external sources:

- SLURP;
- Google Speech Commands v2;
- Mozilla Common Voice 19 English;
- Fluent Speech Commands;
- SNIPS spoken language understanding data;
- Timers and Such;
- MLEnd spoken numerals;
- the Multi-Sensor Voice Command dataset;
- DEMAND and MS-SNSD noise; and
- Filipino-accented supplemental synthetic speech.

The compilation has mixed source licenses and is limited as a whole to
research and education/non-commercial use. Some group recordings and
Filipino-accented synthetic references have additional consent or provenance
caveats. Consult the dataset card and each original source before downloading,
redistributing, or retraining.

#### VCM dataset curation and alterations

- Speaker IDs, including synthetic voice IDs, were split before balancing.
- Real recordings were transcribed with Whisper large-v3-turbo and compared
  with their labels. Incorrect commands, background speech, silence, and
  unsupported slot values were moved to out of scope.
- Unslotted commands containing extra slot information were removed and
  replaced using voices already assigned to the same split.
- Each variation was balanced with real audio first and same-split synthetic
  voices filling shortages.
- Out-of-scope samples mixed noise, Filipino speech, near-miss requests, and
  general speech.
- Synthetic hard negatives included noise-only, 3–6-speaker babble, reversed
  clips, incomplete 25–40% command fragments, and near-silence hiss.
- Supplemental Filipino-accented synthetic data was used only for training.
- Separate tune-out-of-scope voices were excluded from training.

The shipped E8 training run logged 25,941 rows: 10,733 base training rows,
14,120 Filipino-accented supplemental rows, 1,000 generated negatives, and 88
noise rows. Its validation pool contained 4,443 test/tuning rows plus 400
speaker-disjoint tune-out-of-scope examples.

#### VCM preprocessing and selected augmentation

1. Decode and resample to 16 kHz mono.
2. VAD-trim and center a 2.5-second, 40,000-sample window on the speech.
3. Compute a 40-bin power mel spectrogram using a 400-sample FFT/window,
   160-sample hop, and 20–8,000 Hz range.
4. Log-scale, clip to the training range, and apply per-utterance,
   per-mel mean/variance normalization.
5. Rescale the normalized features and supply a 1 × 40 × 250 tensor.

For the selected E8 model, online waveform augmentation used:

- a 50% chance of DEMAND/MS-SNSD room noise at 12–28 dB SNR;
- random gain within ±4 dB; and
- a time shift of up to ±10% of the clip.

Experiments also evaluated speed perturbation, reverberation, babble,
SpecAugment, label smoothing, different channel widths, deeper CNNs, and
different GRU sizes. Those optional alterations were not enabled in the
selected E8_gru_s301 recipe. Real-voice rows, weak twin intents, inverse-class
frequency, and the REJECT intent received additional loss weighting.

#### VCM architecture and optimization

The shipped model is **E8_gru_s301**, a 3,192,807-parameter CRNN:

- convolution blocks with 32, 64, and 128 channels;
- a three-layer, 256-unit bidirectional GRU;
- a shared fully connected representation;
- intent head A: 19 actions plus REJECT; and
- value head B: 18 slot tokens plus REJECT.

It used seed 301, batch size 128, AdamW at 3e-4 with weight decay 1e-4, cosine
learning-rate decay, weighted cross entropy, a maximum of 36 epochs, and early
stopping after six validation epochs without improvement. The best validation
end-to-end score was selected at epoch 26; training stopped at epoch 32.

Recorded single-model evaluation:

| Evaluation | Intent accuracy | End-to-end accuracy |
|---|---:|---:|
| Test, all voices | 91.6% | 90.5% |
| Test, real voices | 65.3% | 63.1% |
| Holdout, all voices | 89.1% | 87.6% |
| Holdout, real voices | 81.2% | 79.2% |
| Separate synthetic-negative test | 79.6% rejected | 79.6% rejected |

The selected checkpoint was exported as an ONNX opset-13 graph with dynamic
batch size. Audio preprocessing remains in **vcm_infer.py** so the Raspberry Pi
runtime needs only NumPy and ONNX Runtime rather than the PyTorch training
stack. The exported graph, exact melbank, and labels are included in
**vcm_model/**.
