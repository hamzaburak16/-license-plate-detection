import os
import re

import cv2
import easyocr
import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from ultralytics import YOLO

st.set_page_config(page_title="Türk Plaka Tanıma Sistemi", page_icon="🚗")

MODEL_YOLU = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_plaka_model.pt")

IZINLI_KARAKTERLER = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# OCR'ın sık karıştırdığı karakterler
HARFTEN_RAKAMA = {"O": "0", "Q": "0", "D": "0", "I": "1", "Z": "2", "S": "5", "G": "6", "B": "8"}
RAKAMDAN_HARFE = {"0": "O", "1": "I", "2": "Z", "5": "S", "6": "G", "8": "B"}


@st.cache_resource(show_spinner="Modeller yükleniyor (ilk açılışta 1-2 dakika sürebilir)...")
def modelleri_yukle():
    model = YOLO(MODEL_YOLU)
    reader = easyocr.Reader(["en"], gpu=False)
    return model, reader


def konuma_gore_duzelt(metin):
    """
    Türk plakası: 2 rakam (il kodu) + 1-3 harf + 2-4 rakam.
    Metnin uzunluğuna uyan her olası bölünmeyi dener, konuma göre karışan
    karakterleri düzeltir ve en az değişiklik gerektireni seçer.
    Hiçbir bölünme uymazsa (None, 0) döndürür.
    """

    def rakama_cevir(parca):
        sonuc, degisim = "", 0
        for k in parca:
            if k.isdigit():
                sonuc += k
            elif k in HARFTEN_RAKAMA:
                sonuc += HARFTEN_RAKAMA[k]
                degisim += 1
            else:
                return None, 0
        return sonuc, degisim

    def harfe_cevir(parca):
        sonuc, degisim = "", 0
        for k in parca:
            if k.isalpha():
                sonuc += k
            elif k in RAKAMDAN_HARFE:
                sonuc += RAKAMDAN_HARFE[k]
                degisim += 1
            else:
                return None, 0
        return sonuc, degisim

    en_iyi = None
    for harf_sayisi in (1, 2, 3):
        for son_rakam_sayisi in (2, 3, 4):
            if 2 + harf_sayisi + son_rakam_sayisi != len(metin):
                continue

            il, d1 = rakama_cevir(metin[:2])
            harfler, d2 = harfe_cevir(metin[2:2 + harf_sayisi])
            rakamlar, d3 = rakama_cevir(metin[2 + harf_sayisi:])

            if il is None or harfler is None or rakamlar is None:
                continue
            if not (1 <= int(il) <= 81):
                continue

            maliyet = d1 + d2 + d3
            aday = il + harfler + rakamlar
            if en_iyi is None or maliyet < en_iyi[1]:
                en_iyi = (aday, maliyet)

    return en_iyi if en_iyi else (None, 0)


def plaka_hazirla(plaka_img):
    """Mavi TR şeridini ve kenarları keser, büyütür, 3 farklı şekilde işler."""
    h, w = plaka_img.shape[:2]

    sol = int(w * 0.10)  # mavi TR şeridi
    pay_x = int(w * 0.02)
    pay_y = int(h * 0.05)
    kesit = plaka_img[pay_y:h - pay_y, sol:w - pay_x]
    if kesit.size == 0:
        kesit = plaka_img

    kw = kesit.shape[1]
    if kw < 300:
        olcek = 300 / kw
        kesit = cv2.resize(kesit, None, fx=olcek, fy=olcek, interpolation=cv2.INTER_CUBIC)

    gri = cv2.cvtColor(kesit, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(gri)
    _, otsu = cv2.threshold(gri, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    def kenarlik(img):
        return cv2.copyMakeBorder(img, 10, 10, 10, 10, cv2.BORDER_REPLICATE)

    return [("gri", kenarlik(gri)), ("clahe", kenarlik(clahe)), ("otsu", kenarlik(otsu))]


def plaka_oku_coklu(plaka_img, reader):
    """Her görüntü sürümünü OCR'dan geçirir, kalıba uyan en iyi sonucu seçer."""
    surumler = plaka_hazirla(plaka_img)
    adaylar = []
    ham_sonuclar = []

    for ad, goruntu in surumler:
        sonuc = reader.readtext(goruntu, allowlist=IZINLI_KARAKTERLER)
        sonuc = sorted(sonuc, key=lambda s: s[0][0][0])  # soldan sağa sırala

        metin = "".join(re.sub(r"[^A-Z0-9]", "", t.upper()) for (_, t, _) in sonuc)
        if metin.startswith("TR"):
            metin = metin[2:]
        guven = float(np.mean([g for (_, _, g) in sonuc])) if sonuc else 0.0

        ham_sonuclar.append(f"{ad}: {metin} (güven {guven:.2f})")

        duzeltilmis, maliyet = konuma_gore_duzelt(metin)
        if duzeltilmis:
            adaylar.append((guven - 0.05 * maliyet, duzeltilmis, guven))

    if adaylar:
        _, en_iyi, en_iyi_guven = max(adaylar)
        if en_iyi_guven < 0.4:
            en_iyi += " (düşük güven)"
    else:
        en_iyi = None

    return en_iyi, ham_sonuclar, surumler[0][1]


def tahmin_et(img_rgb, model, reader):
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    sonuc = model(img_bgr, verbose=False)

    plakalar = []
    debug_metinleri = []
    kesilen = None
    sonuc_img = img_bgr.copy()

    for r in sonuc:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            plaka_img = img_bgr[y1:y2, x1:x2]
            if plaka_img.size == 0:
                continue

            cv2.rectangle(sonuc_img, (x1, y1), (x2, y2), (255, 0, 0), 3)

            plaka, ham_sonuclar, debug_img = plaka_oku_coklu(plaka_img, reader)
            debug_metinleri.extend(ham_sonuclar)
            if kesilen is None:
                kesilen = cv2.cvtColor(debug_img, cv2.COLOR_GRAY2RGB)

            plakalar.append(plaka if plaka else "Okunamadı (kalıba uymuyor)")

    sonuc_rgb = cv2.cvtColor(sonuc_img, cv2.COLOR_BGR2RGB)
    return sonuc_rgb, plakalar, debug_metinleri, kesilen


st.title("🚗 Türk Plaka Tanıma Sistemi")
st.write(
    "Bir araç fotoğrafı yükleyin. Kendi eğittiğim YOLOv8 modeli plakayı bulur, "
    "EasyOCR plaka yazısını okur."
)
st.caption("OCR sonuçları görüntü kalitesine göre hatalı olabilir.")

yuklenen = st.file_uploader("Araç fotoğrafı yükle", type=["jpg", "jpeg", "png"])

if yuklenen is not None:
    model, reader = modelleri_yukle()

    img = Image.open(yuklenen)
    img = ImageOps.exif_transpose(img).convert("RGB")
    # Çok büyük fotoğraflar belleği doldurmasın
    img.thumbnail((1600, 1600))
    img_rgb = np.array(img)

    with st.spinner("Plaka aranıyor..."):
        sonuc_rgb, plakalar, debug_metinleri, kesilen = tahmin_et(img_rgb, model, reader)

    st.image(sonuc_rgb, caption="Tespit Sonucu", use_container_width=True)

    if plakalar:
        for p in plakalar:
            st.success(f"Okunan plaka: **{p}**")
    else:
        st.warning("Plaka bulunamadı.")

    with st.expander("Ayrıntılar (OCR ham sonucu)"):
        for d in debug_metinleri:
            st.text(d)
        if kesilen is not None:
            st.image(kesilen, caption="OCR'a giden görüntü")
