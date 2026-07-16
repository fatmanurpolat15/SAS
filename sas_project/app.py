"""
SAS Mayin Siniflandirici — Streamlit Arayuzu
Calistirmak icin: streamlit run app.py
"""
import os, time
from pathlib import Path
import numpy as np
import cv2
import streamlit as st
from PIL import Image

# Model yolu — train.py'nin ciktigi yere gore guncelle
DEFAULT_WEIGHTS = "runs_sas/exp_v1/weights/best.pt"
WEIGHTS = os.environ.get("SAS_WEIGHTS", DEFAULT_WEIGHTS)

LABEL_TR = {"mayin":"MAYIN","kaya":"KAYA","enkaz":"ENKAZ","diger":"DIGER"}
COLOR    = {"mayin":"#d62728","kaya":"#7f7f7f","enkaz":"#ff7f0e","diger":"#2ca02c"}
ICON     = {"mayin":"💣","kaya":"🪨","enkaz":"🔩","diger":"🌊"}
MARGIN_THRESHOLD = 0.15

@st.cache_resource(show_spinner="Model yukleniyor...")
def load_model(w):
    from ultralytics import YOLO
    p = Path(w)
    if not p.exists():
        # Alternatif yolu dene (ultralytics bazen runs/classify/ ekler)
        alt = Path("runs") / "classify" / w
        if alt.exists():
            p = alt
        else:
            st.error(f"Model bulunamadi: {p.resolve()}")
            st.info("Once run_pipeline.py calistirin.")
            st.stop()
    return YOLO(str(p))

def preprocess_for_sas(pil_img):
    """Gercek fotograflari model egitim verisine benzetir: griye cevir + CLAHE."""
    img = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

def predict(model, pil_img, use_preprocess=True):
    t0 = time.time()
    arr = preprocess_for_sas(pil_img) if use_preprocess else np.array(pil_img.convert("RGB"))
    r = model.predict(arr, verbose=False)[0]
    names = r.names; probs = r.probs
    k = min(4, len(names))
    ids = probs.top5[:k]; confs = probs.top5conf.tolist()[:k]
    return [(names[i], float(c)) for i,c in zip(ids,confs)], (time.time()-t0)*1000

def is_uncertain(preds, margin_th=MARGIN_THRESHOLD):
    return len(preds) >= 2 and (preds[0][1] - preds[1][1]) < margin_th

# ── Sayfa Ayarlari ──
st.set_page_config(page_title="SAS Mayin Siniflandirici", page_icon="🌊", layout="wide")
st.title("🌊 SAS Mayin Siniflandirici")
st.caption("Siniflar: MAYIN · KAYA · ENKAZ · DIGER — YOLOv8n-cls")

with st.sidebar:
    st.header("Ayarlar")
    w_in = st.text_input("Model yolu (.pt)", value=WEIGHTS)
    conf_th = st.slider("Guven esigi", 0.0, 1.0, 0.50, 0.05)
    margin_th = st.slider("Belirsizlik esigi", 0.0, 0.5, 0.15, 0.05)
    topk = st.slider("Top-K goster", 1, 4, 4)
    use_preprocess = st.toggle("SAS on isleme (CLAHE)", value=True,
        help="Renkli fotograflar icin ac. Zaten gri SAS goruntuleri icin kapabilirsin.")
    st.info("Model sentetik gri SAS verisiyle egitildi. On isleme renkli fotograflari normalize eder.")

model = load_model(w_in)
st.sidebar.success("Model yuklendi ✅")

st.markdown("### Sonar Goruntusu Yukle")
uploaded = st.file_uploader("PNG, JPG, BMP, TIF",
    type=["png","jpg","jpeg","bmp","tif","tiff"], accept_multiple_files=True)

if not uploaded:
    st.info("Goruntu yuklemek icin yukardaki alana tikla.")
    st.stop()

for f in uploaded:
    st.markdown("---")
    c1, c2 = st.columns([1, 1.4], gap="large")
    img = Image.open(f)

    with c1:
        st.image(img, caption=f.name, use_container_width=True)
        if use_preprocess:
            proc = preprocess_for_sas(img)
            st.image(Image.fromarray(cv2.cvtColor(proc, cv2.COLOR_BGR2RGB)),
                     caption="On isleme sonrasi", use_container_width=True)

    with c2:
        with st.spinner("Tahmin yapiliyor..."):
            preds, ms = predict(model, img, use_preprocess=use_preprocess)

        top_lbl, top_conf = preds[0]
        uncertain = is_uncertain(preds, margin_th)
        col = COLOR.get(top_lbl, "#1f77b4")
        tr = LABEL_TR.get(top_lbl, top_lbl.upper())
        icon = ICON.get(top_lbl, "?")

        if uncertain:
            st.markdown(f"""<div style="padding:16px;border-radius:12px;
                background:#9467bd18;border:2.5px solid #9467bd;">
              <div style="font-size:13px;color:#888;">Tahmin</div>
              <div style="font-size:32px;font-weight:700;color:#9467bd;">❓ BELIRSIZ — muhtemel {tr}</div>
              <div style="font-size:14px;color:#555;">En yuksek: <b>{top_conf*100:.2f}%</b>
              &nbsp;|&nbsp; 2. sinif: <b>{preds[1][1]*100:.2f}%</b> &nbsp;|&nbsp; {ms:.1f} ms</div>
            </div>""", unsafe_allow_html=True)
            st.warning("⚠️ Model yeterince emin degil. Daha fazla egitim verisi gerekebilir.")
        else:
            low_warn = " DUSUK GUVEN" if top_conf < conf_th else ""
            st.markdown(f"""<div style="padding:16px;border-radius:12px;
                background:{col}18;border:2.5px solid {col};">
              <div style="font-size:13px;color:#888;">Tahmin</div>
              <div style="font-size:36px;font-weight:700;color:{col};">{icon} {tr}{low_warn}</div>
              <div style="font-size:15px;color:#555;">Guven: <b>{top_conf*100:.2f}%</b>
              &nbsp;|&nbsp; {ms:.1f} ms</div>
            </div>""", unsafe_allow_html=True)

            if top_conf >= conf_th:
                msgs = {
                    "mayin": ("error",   "🚨 OLASI MAYIN TESPIT EDILDI."),
                    "kaya":  ("info",    "🪨 Dogal kaya formasyonu."),
                    "enkaz": ("warning", "🔩 Yapisal enkaz / metalik cisim."),
                    "diger": ("success", "🌊 Tehlikeli olmayan dip yapisi."),
                }
                fn_name, msg = msgs.get(top_lbl, ("info", ""))
                getattr(st, fn_name)(msg)

        st.markdown("**📊 Tum Sinif Skorlari:**")
        for lbl, prob in preds[:topk]:
            st.progress(min(max(float(prob),0.0),1.0),
                text=f"{ICON.get(lbl,'')} {LABEL_TR.get(lbl,lbl.upper())}: {prob*100:.2f}%")
