"""
Sentetik SAS goruntu uretici — multiprocessing ile hizli.
Siniflar: mayin, kaya, enkaz, diger
"""
import os, argparse
import numpy as np
import cv2
from pathlib import Path
from multiprocessing import Pool, cpu_count


def make_seabed(size, rng):
    base = rng.normal(110, 18, (size, size)).astype(np.float32)
    xs, ys = np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size))
    wave = 12 * np.sin(2*np.pi*rng.uniform(3,12)*(
        xs*np.cos(rng.uniform(0,np.pi)) + ys*np.sin(rng.uniform(0,np.pi))))
    return cv2.GaussianBlur((base+wave).astype(np.float32), (5,5), 1.5)

def add_speckle(img, rng, intensity=0.18):
    return img*(1-intensity) + img*rng.gamma(2.0, 0.5, img.shape)*intensity

def add_shadow(img, cx, cy, ow, oh, dx, dy, strength, size):
    shadow = np.ones((size, size), np.float32)
    length = int(ow * 1.8)
    for i in range(length):
        t = i / max(length, 1)
        px, py = int(cx+dx*(ow/2+i)), int(cy+dy*(oh/2+i))
        if 0 <= px < size and 0 <= py < size:
            cv2.circle(shadow, (px,py), max(2, int(oh*(0.55-0.25*t))), 1-strength*(1-t), -1)
    return img * cv2.GaussianBlur(shadow, (9,9), 3)

def synth_mayin(size, seed):
    rng = np.random.default_rng(seed)
    img = make_seabed(size, rng)
    cx, cy = int(size/2+rng.normal(0,size*0.08)), int(size/2+rng.normal(0,size*0.08))
    r = int(rng.uniform(size*0.10, size*0.18))
    o = img.copy()
    cv2.ellipse(o,(cx,cy),(r,int(r*0.7)),0,0,360,245,-1)
    cv2.ellipse(o,(cx,cy+int(r*0.15)),(int(r*0.85),int(r*0.45)),0,0,360,190,-1)
    cv2.ellipse(o,(cx-int(r*0.4),cy-int(r*0.3)),(int(r*0.35),int(r*0.2)),
                int(rng.integers(0,180)),0,360,255,-1)
    img = cv2.GaussianBlur(o,(5,5),1.2)
    a = rng.uniform(0, 2*np.pi)
    img = add_shadow(img,cx,cy,r*2,r*1.4,np.cos(a),np.sin(a),rng.uniform(0.45,0.65),size)
    return np.clip(add_speckle(img,rng,rng.uniform(0.15,0.25)),0,255).astype(np.uint8)

def synth_kaya(size, seed):
    rng = np.random.default_rng(seed)
    img = make_seabed(size, rng)
    cx, cy = int(size/2+rng.normal(0,size*0.1)), int(size/2+rng.normal(0,size*0.1))
    s = int(rng.uniform(size*0.12, size*0.22))
    n = int(rng.integers(7,14))
    angles = np.sort(rng.uniform(0,2*np.pi,n))
    radii = rng.uniform(0.6,1.2,n)*s
    pts = np.array([[cx+r*np.cos(a),cy+r*np.sin(a)] for a,r in zip(angles,radii)], dtype=np.int32)
    o = img.copy()
    cv2.fillPoly(o,[pts],200)
    for _ in range(int(rng.integers(8,16))):
        cv2.circle(o,(int(cx+rng.normal(0,s*0.4)),int(cy+rng.normal(0,s*0.4))),
                   int(rng.uniform(2,max(3,s*0.2))),int(rng.uniform(150,230)),-1)
    img = cv2.GaussianBlur(o,(5,5),1.5)
    a = rng.uniform(0, 2*np.pi)
    img = add_shadow(img,cx,cy,s*2,s*1.6,np.cos(a),np.sin(a),rng.uniform(0.35,0.55),size)
    return np.clip(add_speckle(img,rng,rng.uniform(0.20,0.30)),0,255).astype(np.uint8)

def synth_enkaz(size, seed):
    rng = np.random.default_rng(seed)
    img = make_seabed(size, rng)
    cx, cy = int(size/2+rng.normal(0,size*0.08)), int(size/2+rng.normal(0,size*0.08))
    s = int(rng.uniform(size*0.09, size*0.15))
    o = img.copy()
    angle = float(rng.uniform(0,180))
    box = cv2.boxPoints(((cx,cy),(int(s*rng.uniform(2.2,3.2)),int(s*rng.uniform(0.35,0.65))),angle)).astype(np.int32)
    cv2.fillPoly(o,[box],220)
    cv2.polylines(o,[box],True,250,2)
    rad = np.deg2rad(angle)
    for _ in range(int(rng.integers(3,6))):
        t = float(rng.uniform(-0.45,0.45))
        cv2.circle(o,(int(cx+np.cos(rad)*s*2.5*t),int(cy+np.sin(rad)*s*2.5*t)),
                   int(rng.uniform(2,5)),240,-1)
    img = cv2.GaussianBlur(o,(3,3),1.0)
    a = rng.uniform(0, 2*np.pi)
    img = add_shadow(img,cx,cy,s*3.5,s*0.9,np.cos(a),np.sin(a),rng.uniform(0.4,0.6),size)
    return np.clip(add_speckle(img,rng,rng.uniform(0.20,0.30)),0,255).astype(np.uint8)

def synth_diger(size, seed):
    rng = np.random.default_rng(seed)
    img = make_seabed(size, rng)
    if rng.random() < 0.7:
        cx, cy = int(size/2+rng.normal(0,size*0.2)), int(size/2+rng.normal(0,size*0.2))
        s = int(rng.uniform(size*0.06, size*0.14))
        o = img.copy()
        c = int(rng.integers(0,3))
        if c == 0:
            cv2.ellipse(o,(cx,cy),(int(s*1.4),int(s*0.18)),float(rng.uniform(0,180)),0,360,190,-1)
        elif c == 1:
            for _ in range(int(rng.integers(20,40))):
                cv2.circle(o,(int(cx+rng.normal(0,s)),int(cy+rng.normal(0,s*0.7))),
                           int(rng.integers(1,4)),int(rng.uniform(140,210)),-1)
        else:
            cv2.circle(o,(cx,cy),int(s*0.9),int(rng.uniform(150,180)),-1)
        img = cv2.GaussianBlur(o,(3,3),1.0)
    return np.clip(add_speckle(img,rng,rng.uniform(0.25,0.35)),0,255).astype(np.uint8)

GENERATORS = {"mayin":synth_mayin,"kaya":synth_kaya,"enkaz":synth_enkaz,"diger":synth_diger}

def _worker(args):
    cls_name, size, seed, out_path = args
    img = GENERATORS[cls_name](size, seed)
    cv2.imwrite(out_path, cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))

def build_dataset(out_dir, n_per_class=600, img_size=256, train_ratio=0.8, val_ratio=0.1, seed=42):
    rng = np.random.default_rng(seed)
    out = Path(out_dir)
    for split in ("train","val","test"):
        for cls in GENERATORS:
            (out/split/cls).mkdir(parents=True, exist_ok=True)

    n_tr = int(n_per_class*train_ratio)
    n_va = int(n_per_class*val_ratio)
    n_te = n_per_class - n_tr - n_va

    jobs = []
    idx = 0
    for cls in GENERATORS:
        for split, cnt in (("train",n_tr),("val",n_va),("test",n_te)):
            for _ in range(cnt):
                seed_i = int(rng.integers(0, 10_000_000))
                path = str(out/split/cls/f"{cls}_{idx:05d}.png")
                jobs.append((cls, img_size, seed_i, path))
                idx += 1

    workers = max(1, cpu_count()-1)
    print(f"[+] {len(jobs)} goruntu, {workers} CPU core ile uretiliyor...")
    with Pool(workers) as pool:
        done = 0
        for _ in pool.imap_unordered(_worker, jobs, chunksize=20):
            done += 1
            if done % 400 == 0:
                print(f"  {done}/{len(jobs)} tamamlandi...")
    print(f"[OK] Dataset olusturuldu: {out.resolve()}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out",  default="dataset_synth")
    p.add_argument("--n",    type=int, default=600)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    build_dataset(a.out, a.n, a.size, seed=a.seed)
