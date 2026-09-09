import random
import numpy as np
import platform
import psutil
import json
import os
import sys
import datetime
import subprocess

VALID_SEEDS = [42, 123, 456, 789, 2026]

def set_deterministic_seed(seed: int):
    """
    Forces deterministic execution across python random and numpy.
    For sklearn models, the `random_state` should be passed this same seed upon initialization.
    """
    if seed not in VALID_SEEDS:
        raise ValueError(f"Seed {seed} must be one of {VALID_SEEDS} for standard benchmarks.")
    random.seed(seed)
    np.random.seed(seed)
    
def get_git_commit_sha() -> str:
    try:
        sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL).decode('ascii').strip()
        return sha
    except Exception:
        return "UNKNOWN"

def capture_environment_specs(output_path: str):
    """
    Captures system specifications and python environment details for reproducibility.
    """
    try:
        cpu_name = subprocess.check_output("wmic cpu get name", shell=True).decode().split('\n')[1].strip()
    except Exception:
        cpu_name = platform.processor()

    env_specs = {
        "timestamp": datetime.datetime.now().isoformat(),
        "git_commit_sha": get_git_commit_sha(),
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "python_version": sys.version,
        "cpu_name": cpu_name,
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "total_ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "hostname": "ANONYMIZED",
        "packages": {}
    }
    
    # Try to grab key package versions
    try:
        import zstandard
        env_specs["packages"]["zstandard"] = zstandard.__version__
    except ImportError: pass
    
    try:
        import brotli
        env_specs["packages"]["brotli"] = brotli.version
    except ImportError: pass
    
    try:
        import lz4
        env_specs["packages"]["lz4"] = lz4.VERSION
    except ImportError: pass
    
    try:
        import sklearn
        env_specs["packages"]["scikit-learn"] = sklearn.__version__
    except ImportError: pass

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(env_specs, f, indent=4)
