# ---------------------------------------------
# Shared constants for BOTH evaluations
# ---------------------------------------------
MODEL_NAME  = "multi-qa-mpnet-base-cos-v1"
BATCH_SIZE  = 32
K_VALUES    = [10]
SEED        = 42
DEVICE      = "cuda"   # will fall back to cpu if unavailable

# Home‑Depot settings
HOME_QUERY_RATIO = 0.50
HOME_DISTRACTORS = 20_000

# MS‑MARCO settingsms   
MSMARCO_QUERY_RATIO = 0.50
MSMARCO_DISTRACTORS = 100_000