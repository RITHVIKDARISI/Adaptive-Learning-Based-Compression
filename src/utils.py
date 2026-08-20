import os
import sys
import platform
import subprocess
import json

def get_project_root():
    """Returns the absolute path to the project root directory."""
    # This file is in src/, so root is one level up
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_data_dir():
    """Returns the absolute path to the data directory, creating it if necessary."""
    data_dir = os.path.join(get_project_root(), "data")
    os.makedirs(data_dir, exist_ok=True)
    return data_dir

def log_environment_specs():
    """Logs system hardware and software specifications for reproducibility."""
    specs = {
        "os": platform.system(),
        "os_release": platform.release(),
        "os_version": platform.version(),
        "python_version": sys.version,
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "gpu_available": False,
        "gpu_name": None,
        "ram_gb": None
    }
    
    # Try to get RAM size on Windows
    if specs["os"] == "Windows":
        try:
            out = subprocess.check_output("wmic computersystem get totalphysicalmemory", shell=True).decode()
            lines = out.strip().split("\n")
            if len(lines) > 1:
                bytes_ram = int(lines[1].strip())
                specs["ram_gb"] = round(bytes_ram / (1024**3), 2)
        except Exception:
            pass
            
    # Try to see if GPU is available (via nvidia-smi if installed)
    try:
        out = subprocess.check_output("nvidia-smi --query-gpu=name --format=csv,noheader", shell=True).decode()
        specs["gpu_available"] = True
        specs["gpu_name"] = out.strip()
    except Exception:
        pass

    # Try to log package versions if installed
    packages = ["zstandard", "brotli", "lz4", "scikit-learn", "pandas", "matplotlib", "numpy", "datasketch", "datasets"]
    package_versions = {}
    for pkg in packages:
        try:
            # map import names to pip names
            import_name = pkg
            if pkg == "scikit-learn":
                import_name = "sklearn"
            __import__(import_name)
            mod = sys.modules[import_name]
            package_versions[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            package_versions[pkg] = "not installed"
    specs["packages"] = package_versions

    # Save to data/env_specs.json
    output_path = os.path.join(get_data_dir(), "env_specs.json")
    with open(output_path, "w") as f:
        json.dump(specs, f, indent=4)
        
    print(f"Logged environment specs to {output_path}")
    return specs

if __name__ == "__main__":
    log_environment_specs()
