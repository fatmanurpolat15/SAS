"""
Sentetik ve gercek verileri birlestirip train/val/test klasorlerine kopyalar.
"""
import argparse, shutil, random
from pathlib import Path

CLASSES = ("mayin", "kaya", "enkaz", "diger")
EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

def split_real(real_root, ratios=(0.8, 0.1, 0.1), seed=42):
    rng = random.Random(seed)
    out = {s: {c: [] for c in CLASSES} for s in ("train","val","test")}
    if not real_root.exists():
        return out
    for c in CLASSES:
        d = real_root / c
        if not d.exists(): continue
        files = [p for p in d.iterdir() if p.suffix.lower() in EXTS]
        rng.shuffle(files)
        n = len(files)
        n_tr = int(n*ratios[0]); n_va = int(n*ratios[1])
        out["train"][c] = files[:n_tr]
        out["val"][c]   = files[n_tr:n_tr+n_va]
        out["test"][c]  = files[n_tr+n_va:]
    return out

def copy_synth(synth_root, merged):
    if not synth_root.exists():
        print("[!] Sentetik veri klasoru bulunamadi:", synth_root)
        return
    count = 0
    for split in ("train","val","test"):
        for c in CLASSES:
            src = synth_root/split/c
            dst = merged/split/c
            dst.mkdir(parents=True, exist_ok=True)
            if not src.exists(): continue
            for f in src.iterdir():
                if f.suffix.lower() in EXTS:
                    shutil.copy2(f, dst/f"synth_{f.name}")
                    count += 1
    print(f"[OK] {count} sentetik goruntu kopyalandi.")

def copy_real(real_split, merged):
    count = 0
    for split, cm in real_split.items():
        for c, files in cm.items():
            dst = merged/split/c
            dst.mkdir(parents=True, exist_ok=True)
            for f in files:
                shutil.copy2(f, dst/f"real_{f.name}")
                count += 1
    if count:
        print(f"[OK] {count} gercek goruntu kopyalandi.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--synth", default="dataset_synth")
    p.add_argument("--real",  default="dataset_real")
    p.add_argument("--out",   default="dataset_merged")
    p.add_argument("--seed",  type=int, default=42)
    a = p.parse_args()
    merged = Path(a.out)
    if merged.exists(): shutil.rmtree(merged)
    merged.mkdir(parents=True)
    copy_synth(Path(a.synth), merged)
    copy_real(split_real(Path(a.real), seed=a.seed), merged)
    print(f"[DONE] {merged.resolve()}")
