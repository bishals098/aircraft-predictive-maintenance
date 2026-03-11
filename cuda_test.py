import torch
import subprocess

print("=" * 55)
print("   PyTorch + CUDA + cuDNN Diagnostic Script")
print("=" * 55)

# --- PyTorch & CUDA Version Info ---
print(f"\n[INFO] PyTorch Version     : {torch.__version__}")
print(f"[INFO] CUDA Built Version  : {torch.version.cuda}")
print(f"[INFO] cuDNN Version       : {torch.backends.cudnn.version()}")

# --- CUDA Availability ---
cuda_available = torch.cuda.is_available()
print(f"\n[CHECK] CUDA Available      : {cuda_available}")

if not cuda_available:
    print("\n[ERROR] CUDA is NOT available. Possible reasons:")
    print("  - PyTorch not installed with CUDA support")
    print("  - NVIDIA drivers are outdated or missing")
    print("  - CUDA toolkit version mismatch")
    print("  Run: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    exit(1)

# --- cuDNN Status ---
cudnn_enabled = torch.backends.cudnn.enabled
cudnn_benchmark = torch.backends.cudnn.benchmark
print(f"[CHECK] cuDNN Enabled       : {cudnn_enabled}")
print(f"[CHECK] cuDNN Benchmark Mode: {cudnn_benchmark}")

# --- GPU Count & Details ---
gpu_count = torch.cuda.device_count()
print(f"\n[INFO] Total GPUs Detected  : {gpu_count}")

for i in range(gpu_count):
    props = torch.cuda.get_device_properties(i)
    print(f"\n  --- GPU {i}: {props.name} ---")
    print(f"    Compute Capability     : {props.major}.{props.minor}")
    print(f"    Total VRAM             : {props.total_memory / 1024**3:.2f} GB")
    print(f"    Multiprocessors (SMs)  : {props.multi_processor_count}")
    # RTX 3050 has 128 CUDA cores per SM
    cuda_cores = props.multi_processor_count * 128
    print(f"    CUDA Cores (est.)      : {cuda_cores}  [SM count × 128 cores/SM for Ampere]")
    print(f"    Max Threads/Block      : {props.max_threads_per_block}")
    print(f"    Warp Size              : {props.warp_size}")
    print(f"    L2 Cache Size          : {props.L2_cache_size / 1024:.0f} KB")

# --- Set Active Device ---
torch.cuda.set_device(0)
current = torch.cuda.current_device()
print(f"\n[INFO] Active GPU Index     : {current} ({torch.cuda.get_device_name(current)})")

# --- Functional Test: Tensor on GPU ---
print("\n[TEST] Running GPU tensor computation...")
try:
    a = torch.randn(1000, 1000, device="cuda")
    b = torch.randn(1000, 1000, device="cuda")
    c = torch.matmul(a, b)
    torch.cuda.synchronize()
    print(f"[PASS] Matrix multiply (1000×1000) on GPU succeeded!")
    print(f"       Result tensor shape  : {c.shape}, device: {c.device}")
except Exception as e:
    print(f"[FAIL] GPU computation failed: {e}")

# --- Memory Stats ---
print("\n[INFO] GPU Memory Usage:")
allocated = torch.cuda.memory_allocated(0) / 1024**2
reserved  = torch.cuda.memory_reserved(0)  / 1024**2
print(f"    Allocated               : {allocated:.2f} MB")
print(f"    Reserved (cached)       : {reserved:.2f} MB")

# --- nvidia-smi Summary ---
print("\n[INFO] nvidia-smi GPU Summary:")
try:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,temperature.gpu,utilization.gpu,memory.used,memory.total",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True
    )
    for line in result.stdout.strip().split("\n"):
        parts = [p.strip() for p in line.split(",")]
        labels = ["GPU Name", "Driver Ver", "Temp (°C)", "Util (%)", "Mem Used (MB)", "Mem Total (MB)"]
        for label, val in zip(labels, parts):
            print(f"    {label:<18}: {val}")
except FileNotFoundError:
    print("    [WARN] nvidia-smi not found in PATH.")

print("\n" + "=" * 55)
print("   Diagnostic Complete")
print("=" * 55)