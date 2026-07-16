"""
Tam pipeline: Veri uret → Hazirla → Egit
VS Code terminalinde calistir: python run_pipeline.py
"""
import os, sys, glob, time, shutil, subprocess
from pathlib import Path

BASE = Path(__file__).parent.resolve()

# ── GPU kontrolu ──────────────────────────────
try:
    import torch
    DEVICE = "0" if torch.cuda.is_available() else "cpu"
    if DEVICE == "0":
        print(f"✅ GPU bulundu: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️  GPU bulunamadi — CPU ile calisacak.")
except ImportError:
    print("❌ torch yuklu degil. Once calistir: pip install -r requirements.txt")
    sys.exit(1)

t_start = time.time()

def run(cmd):
    print(f"\n$ {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=str(BASE))
    if result.returncode != 0:
        print(f"❌ Komut basarisiz: {cmd}")
        sys.exit(1)

# 1) Temizle
print("\n" + "="*55)
print("[1/4] Temizleniyor...")
print("="*55)
for d in ["dataset_synth", "dataset_merged", "runs_sas"]:
    p = BASE / d
    if p.exists():
        shutil.rmtree(p)
        print(f"  Silindi: {d}")

# 2) Veri uret
print("\n" + "="*55)
print("[2/4] Sentetik veri uretiliyor (600 goruntu/sinif)...")
print("="*55)
run("python generate_synthetic.py --out dataset_synth --n 600 --size 256 --seed 42")

# 3) Birlestir
print("\n" + "="*55)
print("[3/4] Dataset hazirlaniyor...")
print("="*55)
run("python prepare_dataset.py --synth dataset_synth --out dataset_merged")

# 4) Egit
batch = 64 if DEVICE == "0" else 16
print("\n" + "="*55)
print(f"[4/4] Egitim basliyor (device={DEVICE}, batch={batch})...")
print("="*55)
run(
    f"python train.py "
    f"--data dataset_merged "
    f"--model yolov8n-cls.pt "
    f"--epochs 50 "
    f"--imgsz 256 "
    f"--batch {batch} "
    f"--device {DEVICE} "
    f"--patience 20 "
    f"--project runs_sas "
    f"--name exp_v1"
)

# Model bul — proje klasoru altinda ara (train.py artik buraya kaydediyor)
MODEL_PATH = None
search_paths = [
    BASE / "runs_sas" / "exp_v1" / "weights" / "best.pt",
]
# Glob ile de ara
for found in glob.glob(str(BASE / "**" / "best.pt"), recursive=True):
    search_paths.append(Path(found))

for p in search_paths:
    if Path(p).exists():
        MODEL_PATH = str(Path(p).resolve())
        break

elapsed = time.time() - t_start
print(f"\nToplam sure: {elapsed/60:.1f} dakika")

if MODEL_PATH:
    (BASE / "model_path.txt").write_text(MODEL_PATH)
    print(f"\n✅ Model hazir: {MODEL_PATH}")
    print("\n" + "="*55)
    print("SIRADAKI ADIM — Arayuzu baslat:")
    print("  python -m streamlit run app.py")
    print("="*55)
else:
    # Egitim loglarindan yolu bul ve goster
    print("\n❌ Model proje klasorunde bulunamadi.")
    print("   Egitim ciktisindaki MODEL_PATH satirina bakin")
    print("   ve asagidaki komutu o yol ile calistirin:")
    print("\n   set SAS_WEIGHTS=<model_yolu>")
    print("   python -m streamlit run app.py")