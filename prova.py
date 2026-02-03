"""
debug_10_features.py
====================
Run this from your project root:   python debug_10_features.py

What it does:
  1. Loads a 50 000-row sample (fast)
  2. Calls your Preprocessor with checkpoint prints inside _encode_categorical
  3. Diagnoses exactly which feature group vanished and why
  4. Prints the correct fix at the end
"""

import pandas as pd
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
import src.config as cfg
from src.preprocessing import Preprocessor

# ──────────────────────────────────────────────────────────────────────────────
# Load a sample — full dataset not needed for diagnosis
# ──────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print(" LOADING 50k rows for fast diagnosis")
print("=" * 65)
df = pd.read_csv(cfg.TRAIN_DATA, nrows=50_000, low_memory=False)

y = df[cfg.TARGET_COLUMN].copy()
X = df.drop(columns=[cfg.TARGET_COLUMN])

# Expected feature groups (read from your config)
TE_COLS  = getattr(cfg, 'TARGET_ENCODE_FEATURES', [])
OHE_COLS = getattr(cfg, 'ONEHOT_FEATURES',        [])
NUM_COLS = X.select_dtypes(include='number').columns.tolist()

print(f"\n  config.TARGET_ENCODE_FEATURES ({len(TE_COLS)}): {TE_COLS}")
print(f"  config.ONEHOT_FEATURES       ({len(OHE_COLS)}): {OHE_COLS}")
print(f"  numeric columns in X         ({len(NUM_COLS)}): {NUM_COLS}")

# ──────────────────────────────────────────────────────────────────────────────
# Monkey-patch: wrap _encode_categorical with checkpoints
# ──────────────────────────────────────────────────────────────────────────────
_orig = Preprocessor._encode_categorical

def _debug_encode(self, X, y=None):
    print(f"\n  [encode] INPUT  -> shape {X.shape}, cols = {list(X.columns)}")
    result = _orig(self, X, y)                         # call YOUR code as-is
    print(f"  [encode] OUTPUT -> shape {result.shape}, cols = {list(result.columns)}\n")
    return result

Preprocessor._encode_categorical = _debug_encode

# ──────────────────────────────────────────────────────────────────────────────
# Run the pipeline (handles both common call signatures)
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print(" RUNNING Preprocessor")
print("=" * 65)

prep = Preprocessor()
df_sample = pd.concat([X, y], axis=1)

try:
    result = prep.fit_transform(df_sample)
    # fit_transform may return a tuple (train/val/test splits) or a single df
    if isinstance(result, tuple):
        X_out = result[0]   # X_train
    else:
        X_out = result
except Exception as e:
    print(f"  [!] fit_transform failed ({e}) -- trying fit + transform ...")
    prep.fit(X, y)
    X_out = prep.transform(X)

final_cols = list(X_out.columns)

# ──────────────────────────────────────────────────────────────────────────────
# Diagnosis
# ──────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print(" DIAGNOSIS")
print("=" * 65)

te_present  = [c for c in TE_COLS  if c in final_cols]
ohe_present = [c for c in final_cols if c not in TE_COLS and c not in NUM_COLS]
num_present = [c for c in NUM_COLS  if c in final_cols]

print(f"\n  Final feature count          : {len(final_cols)}")
print(f"  +-- Target-encoded present   : {len(te_present)}  {te_present}")
print(f"  +-- OHE-expanded present     : {len(ohe_present)}")
print(f"  +-- Original numeric present : {len(num_present)}  {num_present}")

# ──────────────────────────────────────────────────────────────────────────────
# Root cause + fix
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print(" ROOT CAUSE  &  FIX")
print("=" * 65)

if len(ohe_present) == 0:
    print("""
  [X]  Zero OHE columns in the output.
       pd.get_dummies() was called but its result was never joined
       back into the main DataFrame.  The 15 original OHE columns
       (object dtype) were also lost because get_dummies consumed
       them without replacing them in X.

  ── The broken pattern (what you likely have) ──────────────────
      ohe_df = pd.get_dummies(X[OHE_COLS], columns=OHE_COLS)
      X = ohe_df                              # <-- replaces X entirely

      # OR simply forgot to use the result:
      pd.get_dummies(X, columns=OHE_COLS)     # <-- return value discarded
  ─────────────────────────────────────────────────────────────────

  ── The correct 3-step pattern ─────────────────────────────────
      # 1. One-hot encode ONLY the OHE columns
      ohe_df = pd.get_dummies(X[OHE_COLS], columns=OHE_COLS, dtype=int)

      # 2. Drop the original OHE columns from X (they are now in ohe_df)
      X = X.drop(columns=OHE_COLS)

      # 3. Concat: everything that survived + the new dummy columns
      X = pd.concat([X, ohe_df], axis=1)
  ─────────────────────────────────────────────────────────────────

  Open preprocessing.py  ->  find _encode_categorical
  Replace the pd.get_dummies section with the 3 lines above.
  Then re-run main.py.  X_train shape should jump from 10 to ~64.
""")

elif len(num_present) == 0:
    print("""
  [X]  OHE columns exist but ALL original numeric features are gone.
       Your concat probably only included the OHE result and dropped
       everything else.

  Fix: ensure the concat keeps the non-OHE part of X:
      X = pd.concat([X.drop(columns=OHE_COLS), ohe_df], axis=1)
""")

else:
    print(f"  [OK] All three groups present -- feature count looks healthy.")
    print(f"       Total: {len(final_cols)} features")