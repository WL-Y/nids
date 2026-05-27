"""
Global configuration: shared parameters for all modules.
Random seed is managed centrally for reproducibility.

Data version: NF-UQ-NIDS-v3
Main training set: NF-UNSW-NB15-v3 (2,365,424 rows x 55 cols)
Target columns: Label = binary (0=Benign), Attack = multi-class strings (10 classes)
"""

import random
import numpy as np

# ==================== Random Seed ====================
SEED = 42


def set_seed(seed=SEED):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


# ==================== Data Paths ====================
DATA_DIR = "data/"
UNSW_PATH = DATA_DIR + "NF-UNSW-NB15-v3.csv"            # main training set, 2.37M rows
CICIDS_PATH = DATA_DIR + "NF-CSE-CIC-IDS2018-v3.csv"    # Stress B target domain, 20.1M rows
BOTIOT_PATH = DATA_DIR + "NF-BoT-IoT-v3.csv"            # Stress B second target domain (optional)
TONIOT_PATH = DATA_DIR + "NF-ToN-IoT-v3.csv"            # Stress B third target domain (optional)

# ==================== Preprocessing Parameters ====================
TEST_SIZE = 0.2
VAL_SIZE = 0.1            # -> train 0.7 / val 0.1 / test 0.2
RANDOM_STATE = SEED

# ==================== Column Definitions ====================
# Columns to drop: timestamps + IP addresses (identifiers, not generalizable features)
DROP_COLUMNS = [
    "FLOW_START_MILLISECONDS", "FLOW_END_MILLISECONDS",
    "IPV4_SRC_ADDR", "IPV4_DST_ADDR",
]

# Categorical features: protocol-related, must be encoded
# PROTOCOL: integer {1=ICMP, 6=TCP, 17=UDP} -> OrdinalEncoder
# L7_PROTO: application-layer protocol, contains non-integer values, must be cleaned first -> OrdinalEncoder
CATEGORICAL_FEATURES = ["PROTOCOL", "L7_PROTO"]

# Numeric features: numeric columns after excluding categorical and drop columns
# Final modeling feature list will be saved to feature_columns.json
NUMERIC_FEATURES = [
    "L4_SRC_PORT", "L4_DST_PORT",
    "IN_BYTES", "IN_PKTS", "OUT_BYTES", "OUT_PKTS",
    "TCP_FLAGS", "CLIENT_TCP_FLAGS", "SERVER_TCP_FLAGS",
    "FLOW_DURATION_MILLISECONDS", "DURATION_IN", "DURATION_OUT",
    "MIN_TTL", "MAX_TTL",
    "LONGEST_FLOW_PKT", "SHORTEST_FLOW_PKT",
    "MIN_IP_PKT_LEN", "MAX_IP_PKT_LEN",
    "SRC_TO_DST_SECOND_BYTES", "DST_TO_SRC_SECOND_BYTES",
    "RETRANSMITTED_IN_BYTES", "RETRANSMITTED_IN_PKTS",
    "RETRANSMITTED_OUT_BYTES", "RETRANSMITTED_OUT_PKTS",
    "SRC_TO_DST_AVG_THROUGHPUT", "DST_TO_SRC_AVG_THROUGHPUT",
    "NUM_PKTS_UP_TO_128_BYTES", "NUM_PKTS_128_TO_256_BYTES",
    "NUM_PKTS_256_TO_512_BYTES", "NUM_PKTS_512_TO_1024_BYTES",
    "NUM_PKTS_1024_TO_1514_BYTES",
    "TCP_WIN_MAX_IN", "TCP_WIN_MAX_OUT",
    "ICMP_TYPE", "ICMP_IPV4_TYPE",
    "DNS_QUERY_ID", "DNS_QUERY_TYPE", "DNS_TTL_ANSWER",
    "FTP_COMMAND_RET_CODE",
    "SRC_TO_DST_IAT_MIN", "SRC_TO_DST_IAT_MAX",
    "SRC_TO_DST_IAT_AVG", "SRC_TO_DST_IAT_STDDEV",
    "DST_TO_SRC_IAT_MIN", "DST_TO_SRC_IAT_MAX",
    "DST_TO_SRC_IAT_AVG", "DST_TO_SRC_IAT_STDDEV",
]

# Protocol-specific features (fill missing with 0, since 0 = protocol not applicable)
PROTOCOL_FEATURES = [
    "ICMP_TYPE", "ICMP_IPV4_TYPE",
    "DNS_QUERY_ID", "DNS_QUERY_TYPE", "DNS_TTL_ANSWER",
    "FTP_COMMAND_RET_CODE",
]

# Exclude PROTOCOL_FEATURES from NUMERIC_FEATURES
NUMERIC_FEATURES = [c for c in NUMERIC_FEATURES if c not in PROTOCOL_FEATURES]

# Target columns: Label = binary, Attack = multi-class (v3 differs from omar.md description)
LABEL_COL = "Label"       # binary: 0=Benign, 1=Attack
ATTACK_COL = "Attack"     # multi-class target: "Benign", "DoS", "Exploits", ...

# ==================== Model Parameters (grid search ranges) ====================
LR_PARAMS = {
    "C": [0.01, 0.1, 1, 10],
    "penalty": ["l2"],
    "solver": ["lbfgs"],
    "max_iter": [3000],
}
RF_PARAMS = {
    "n_estimators": [100, 200],
    "max_depth": [10, 20, None],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
    "class_weight": ["balanced"],
}
XGB_PARAMS = {
    "n_estimators": [100, 200],
    "max_depth": [6, 10],
    "learning_rate": [0.01, 0.1],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
}

# ==================== Stress Test Parameters ====================
# Strategy: first group = 3 rarest classes, second group = 3 medium-frequency classes
HELD_OUT_CLASSES_SETS = [
    ["Worms", "Analysis", "Shellcode"],       # rarest 3 classes
    ["Backdoor", "DoS", "Fuzzers"],            # medium-frequency 3 classes
]

GAUSSIAN_NOISE_STDS = [0.1, 0.5, 1.0]
MASKING_RATES = [0.1, 0.25, 0.5]
FEATURE_DROPOUT_COUNTS = [2, 4, 6]            # drop by feature importance rank
SIGNIFICANCE_RUNS = 5                         # number of repetitions for statistical significance
SIGNIFICANCE_SEEDS = [42, 123, 456, 789, 1111]

# Stress B uses full CICIDS target-domain evaluation with chunked reading.
STRESS_B_CHUNK_SIZE = 200_000

# ==================== Robustness Strategy Parameters ====================
TAU_RANGE = (0.5, 0.99, 0.01)    # start, stop, step
ENSEMBLE_SIZE = 5
ENSEMBLE_MODELS = ["rf", "rf", "xgb", "lr", "rf"]  # heterogeneous ensemble
