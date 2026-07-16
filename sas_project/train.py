"""
YOLOv8 siniflandirma egitimi.
Model proje klasorune kaydedilir (kullanici dizinine degil).
"""
import argparse, json
from pathlib import Path
from ultralytics import YOLO


def main(a):
    dp = Path(a.data).resolve()
    if not dp.exists():
        raise FileNotFoundError(f"Dataset bulunamadi: {dp}")

    # Proje yolunu mutlak yap — boylece Ultralytics her zaman
    # proje klasorune kaydeder, C:\Users\...\runs altina degil
    project_abs = str(Path(a.project).resolve())

    model = YOLO(a.model)
    model.train(
        data=str(dp),
        epochs=a.epochs,
        imgsz=a.imgsz,
        batch=a.batch,
        project=project_abs,   # <-- mutlak yol
        name=a.name,
        patience=a.patience,
        lr0=a.lr0,
        lrf=0.01,
        warmup_epochs=3,
        optimizer=a.optimizer,
        device=a.device,
        verbose=True,
        pretrained=True,
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.3,
        degrees=20.0, translate=0.1, scale=0.3,
        fliplr=0.5, flipud=0.3,
        mosaic=0.5, mixup=0.1,
        erasing=0.0,
    )

    # Gercek kayit dizini
    save_dir = Path(model.trainer.save_dir)
    best = save_dir / "weights" / "best.pt"
    print(f"\n[OK] Model kaydedildi: {best.resolve()}")

    if (dp / "test").exists():
        print("\n[+] Test seti degerlendirme...")
        model.val(data=str(dp), split="test", imgsz=a.imgsz)

    train_dir = dp / "train"
    if train_dir.exists():
        names = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
        save_dir.mkdir(parents=True, exist_ok=True)
        (save_dir / "classes.json").write_text(
            json.dumps({"classes": names}, indent=2)
        )
        print(f"[OK] Siniflar: {names}")

    print(f"\n{'='*55}")
    print(f"MODEL_PATH = \"{best.resolve()}\"")
    print(f"{'='*55}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data",      default="dataset_merged")
    p.add_argument("--model",     default="yolov8n-cls.pt")
    p.add_argument("--epochs",    type=int,   default=50)
    p.add_argument("--imgsz",     type=int,   default=256)
    p.add_argument("--batch",     type=int,   default=16)
    p.add_argument("--lr0",       type=float, default=0.001)
    p.add_argument("--optimizer", default="AdamW")
    p.add_argument("--patience",  type=int,   default=20)
    p.add_argument("--device",    default="cpu")
    p.add_argument("--project",   default="runs_sas")
    p.add_argument("--name",      default="exp_v1")
    main(p.parse_args())