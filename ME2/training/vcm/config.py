"""Shared config for the golden-dataset retrain.

Single source of truth for the schema = vcm_action_runner/intents.py
(19 intents: 13 fixed + 6 slotted, 3 slot values each). The HF dataset
`airimonda/ai231-me2-voice-commands` `command` column already uses exactly
these names, so the mapping is identity.

Architecture replicates the old two-head CNN from
vcm_work/model_twohead/vcm_twohead_best_ai231.pt (1.62M params):
  input  (1, 40, 150)  log-mel spectrogram
  head_a -> 19 intents + 1 REJECT  = 20
  head_b -> 18 slot tokens + 1 REJECT = 19
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
HF_DIR = os.environ.get(
    "AI231_VCM_DATA",
    os.path.join(HERE, "data", "ai231-me2-voice-commands"),
)
DATA_DIR = os.path.join(HF_DIR, "data")
OUT_DIR = os.environ.get("AI231_VCM_OUTPUT", os.path.join(HERE, "checkpoints"))
NOISE_DIR = os.environ.get("AI231_VCM_NOISE", os.path.join(HERE, "noise"))
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------- audio ----
SR = 16000
N_MELS = 40
N_FFT = 400          # 25 ms @ 16 kHz
HOP = 160            # 10 ms @ 16 kHz  -> 150 frames ~= 1.5 s window
WIN = 400
N_FRAMES = 150
FMIN = 20.0
FMAX = 8000.0

# --------------------------------------------------------------- schema ----
FIXED = [
    "PLAY_MUSIC", "WEATHER", "TIME", "LIGHT_ON", "LIGHT_OFF", "PAUSE",
    "STOP", "NEXT", "VOLUME_UP", "VOLUME_DOWN", "CALL", "MESSAGE",
    "LIST_REMINDERS",
]
SLOTTED = {
    "TIMER":           ["10 seconds", "30 seconds", "1 minute"],
    "ALARM":           ["6:00 AM", "8:00 AM", "9:00 PM"],
    "TEMPERATURE":     ["18 degrees", "22 degrees", "26 degrees"],
    "BRIGHTNESS":      ["20 percent", "60 percent", "100 percent"],
    "COLOR":           ["Red", "Blue", "Green"],
    "CREATE_REMINDER": ["Drink water", "Study", "Exercise"],
}
INTENTS = FIXED + list(SLOTTED)          # 19
N_INTENT = len(INTENTS)                  # 19

VALUE_TOKENS = [f"{it}_{v.replace(' ', '_').upper()}"
                for it, vals in SLOTTED.items() for v in vals]
N_VALUE = len(VALUE_TOKENS)              # 18

REJECT = "REJECT"
N_A = N_INTENT + 1                       # 20 (intent + reject)
N_B = N_VALUE + 1                        # 19 (slot token + reject)

INTENT2IDX = {n: i for i, n in enumerate(INTENTS)}
VALUE2IDX = {t: i for i, t in enumerate(VALUE_TOKENS)}
PAIR2TOKEN = {(it, v): f"{it}_{v.replace(' ', '_').upper()}"
              for it, vals in SLOTTED.items() for v in vals}

# ------------------------------------------------------------- train cfg ----
EPOCHS = 24
BATCH = 128
LR = 3e-4
WEIGHT_DECAY = 1e-4
SEED = 13
NUM_WORKERS = 0
THREADS = 32

TRAIN_SPLIT = "train"
VAL_SPLIT = "test"
