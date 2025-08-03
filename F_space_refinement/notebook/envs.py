import os
import subprocess

def set_notebook_env_ids(cuda_version="12.5"):
    # Save original PATH and LD_LIBRARY_PATH
    original_path = os.environ.get("PATH", "")
    original_ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")

    # Set CUDA environment variables
    os.environ["CUDA_HOME"] = f"/usr/local/cuda-{cuda_version}"
    os.environ["PATH"] = f"/usr/local/cuda-{cuda_version}/bin:{original_path}"
    os.environ["LD_LIBRARY_PATH"] = f"/usr/local/cuda-{cuda_version}/lib64:{original_ld_library_path}"

    print(f"CUDA environment variables set for CUDA {cuda_version}")

    # Query GPU type
    try:
        gpu_name = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            encoding="utf-8"
        ).strip()
    except Exception as e:
        print(f"Error querying GPU: {e}")
        return

    # GPU to arch mapping
    arch_map = {
        "P100": "6.0",
        "V100": "7.0",
        "A100": "8.0",
        "A40": "8.6",
        "L40": "8.9",
    }

    # Set TORCH_CUDA_ARCH_LIST
    torch_arch = None
    for key, arch in arch_map.items():
        if key in gpu_name:
            torch_arch = arch
            break

    if torch_arch is None:
        print(f"Unknown GPU: {gpu_name}")
        return

    os.environ["TORCH_CUDA_ARCH_LIST"] = torch_arch
    print(f"Set TORCH_CUDA_ARCH_LIST to {torch_arch}")

def change_exp_dir(CODE_DIR):
    # Change to CODE_DIR
    os.chdir(CODE_DIR)
    notebook_path = os.getcwd()
    print('Current working directory is:\n', notebook_path)
