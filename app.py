import streamlit as st
import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Stajyer Simülatörü - Realistik", layout="wide", page_icon="🎓")

# --- CSS ---
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        font-weight: bold;
    }
    div[data-testid="stMetricValue"] {
        font-size: 24px;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🎓 Stajyer Yerleştirme Simülasyonu")
st.caption("Algoritma + Mülakat/Rededilme Simülasyonu")

# --- İMPORTLAR ---
# Dosyalar yoksa hata vermemesi için try-except bloğu
try:
    import veri_olustur
    import algo_greedy
    import algo_heuristic_hill_climbing
    import algo_heuristic_annealing
except ImportError as e:
    st.error(f"Hata: Gerekli Python dosyaları eksik! Lütfen 'veri_olustur.py' ve algoritma dosyalarının aynı klasörde olduğundan emin olun.\nDetay: {e}")
    st.stop()

# --- SESSION STATE (Hafıza) ---
if 'ogrenciler' not in st.session_state:
    st.session_state['ogrenciler'] = pd.DataFrame()
if 'firmalar' not in st.session_state:
    st.session_state['firmalar'] = pd.DataFrame()
if 'analiz_sonuclari' not in st.session_state:
    st.session_state['analiz_sonuclari'] = {}


# --- PUAN HESAPLA ---
def puan_hesapla(df):
    if df.empty or 'Yerleştiği_Firma' not in df.columns: return 0
    puan_tablosu = {1: 100, 2: 85, 3: 70, 4: 50, 5: 30}
    toplam = 0
    for _, row in df[df['Yerleştiği_Firma'].notna()].iterrows():
        yf = row['Yerleştiği_Firma']
        for i in range(1, 6):
            col_name = f'Tercih{i}'
            if col_name in row and row[col_name] == yf:
                toplam += puan_tablosu.get(i, 10)
                break
    return toplam


# --- GERÇEKÇİ REDDEDİLME FONKSİYONU ---
def mulakat_simulasyonu(df_ogrenciler, df_firmalar, reddetme_orani):
    """
    Algoritma yerleştirdikten sonra firmalar bazı öğrencileri reddeder.
    """
    if reddetme_orani <= 0:
        return df_ogrenciler, df_firmalar, 0

    df_sonuc = df_ogrenciler.copy()
    reddedilen_sayisi = 0

    for idx, row in df_sonuc.iterrows():
        firma = row['Yerleştiği_Firma']
        if pd.notna(firma):
            # Zar at: Eğer gelen sayı orandan küçükse REDDET
            zar = np.random.randint(0, 100)
            if zar < reddetme_orani:
                # Öğrenciyi kov
                df_sonuc.at[idx, 'Yerleştiği_Firma'] = None

                # Firmanın kontenjanını geri ver (Boşa çıktı)
                # 'Firma' sütunu kontrolü
                if 'Firma' in df_firmalar.columns:
                    f_idx = df_firmalar[df_firmalar['Firma'] == firma].index
                    if not f_idx.empty:
                        df_firmalar.at[f_idx[0], 'Kalan_Kontenjan'] += 1
                
                reddedilen_sayisi += 1

    return df_sonuc, df_firmalar, reddedilen_sayisi


# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Kontrol Paneli")

    st.subheader("1. Temel Ayarlar")
    ogr_sayisi = st.number_input("Öğrenci Sayısı", 10, 2000, 150)
    firma_sayisi = st.number_input("Firma Sayısı", 5, 500, 40)

    st.write("---")
    st.subheader("2. Gerçekçilik Ayarları")
    st.info("Algoritma yerleştirse bile, firmalar mülakatta eleyebilir.")
    
    red_orani = st.slider("🚫 Firma Seçiciliği (Reddetme %)", 0, 50, 10,
                          help="0: Herkesi kabul et\n20: %20 ihtimalle reddet")

    st.divider()

    if st.button("🎲 Veri Seti Oluştur", type="primary"):
        # Eşit veri oluştur
        d1, d2 = veri_olustur.veri_seti_olustur(ogr_sayisi, firma_sayisi)

        # Veri setinden dönen df sırasını kontrol et (Hangisi firma hangisi öğrenci?)
        if 'Firma' in d1.columns:
            firmalar_df, ogrenciler_df = d1, d2
        else:
            ogrenciler_df, firmalar_df = d1, d2

        # Sütun İsimlerini Standartlaştır (Hata Önleyici)
        mapping = {'Ortalama': 'GNO', 'Not': 'GNO', 'Puan': 'GNO', 'gno': 'GNO', 
                   'Ogrenci_No': 'Öğrenci', 'Ogrenci': 'Öğrenci'}
        ogrenciler_df.rename(columns=mapping, inplace=True)
        
        if 'Yerleştiği_Firma' not in ogrenciler_df.columns:
            ogrenciler_df['Yerleştiği_Firma'] = None

        st.session_state['ogrenciler'] = ogrenciler_df
        st.session_state['firmalar'] = firmalar_df
        st.session_state['analiz_sonuclari'] = {}

        st.success(f"Veri Hazır! {len(ogrenciler_df)} Öğrenci, {firmalar_df['Kontenjan'].sum()} Kontenjan.")

    st.subheader("Algoritma Başlat")
    btn_greedy = st.button("🚀 Greedy")
    btn_hill = st.button("⛰️ Hill Climbing")
    btn_anneal = st.button("🔥 Annealing")
    st.divider()
    btn_analiz = st.button("📊 Analiz")

    if st.button("🔄 Sıfırla"):
        st.session_state.clear()
        st.rerun()

# --- ANA EKRAN ---
if st.session_state['ogrenciler'].empty:
    st.warning("👈 Lütfen sol menüden önce 'Veri Seti Oluştur' butonuna basın.")
    st.stop()

islem = False
algo = ""
sure = 0
reddedilen_kisi = 0

# --- ALGORİTMA MANTIĞI ---
if btn_greedy:
    algo = "Greedy"
    t1 = time.time()
    res = algo_greedy.greedy_atama(st.session_state['ogrenciler'], st.session_state['firmalar'])

    temp_ogr = res[0] if isinstance(res, tuple) else res
    temp_firma = st.session_state['firmalar']

    # MÜLAKAT SİMÜLASYONU
    final_ogr, final_firma, reddedilen_kisi = mulakat_simulasyonu(temp_ogr, temp_firma, red_orani)

    st.session_state['ogrenciler'] = final_ogr
    st.session_state['firmalar'] = final_firma

    sure = time.time() - t1
    islem = True

elif btn_hill:
    algo = "Hill Climbing"
    t1 = time.time()
    pb = st.progress(0)

    def prog(i):
        if i % 100 == 0:
            time.sleep(0.0005)
            pb.progress(min(i / 3000, 1.0))

    try:
        if hasattr(algo_heuristic_hill_climbing, 'heuristic_atama'):
            func = algo_heuristic_hill_climbing.heuristic_atama
        elif hasattr(algo_heuristic_hill_climbing, 'hill_climbing_main'):
            func = algo_heuristic_hill_climbing.hill_climbing_main
        else:
            func = algo_heuristic_hill_climbing.hill_climbing

        res = func(st.session_state['ogrenciler'], st.session_state['firmalar'], iterasyon=3000, step_callback=prog)

        temp_ogr = res[0] if isinstance(res, tuple) else res
        temp_firma = res[1] if isinstance(res, tuple) else st.session_state['firmalar']

        # MÜLAKAT SİMÜLASYONU
        final_ogr, final_firma, reddedilen_kisi = mulakat_simulasyonu(temp_ogr, temp_firma, red_orani)

        st.session_state['ogrenciler'] = final_ogr
        st.session_state['firmalar'] = final_firma

    except Exception as e:
        st.error(f"Hill Climbing Hatası: {e}")
        st.stop()
    pb.empty()
    sure = time.time() - t1
    islem = True

elif btn_anneal:
    algo = "Simulated Annealing"
    t1 = time.time()
    pb = st.progress(0)

    def prog(i):
        if i % 100 == 0:
            time.sleep(0.0005)
            pb.progress(min(i / 10000, 1.0))

    try:
        if hasattr(algo_heuristic_annealing, 'heuristic_atama'):
            func = algo_heuristic_annealing.heuristic_atama
        elif hasattr(algo_heuristic_annealing, 'simulated_annealing_main'):
            func = algo_heuristic_annealing.simulated_annealing_main
        else:
            func = algo_heuristic_annealing.simulated_annealing

        res = func(st.session_state['ogrenciler'], st.session_state['firmalar'], iterasyon=10000, step_callback=prog)

        temp_ogr = res[0] if isinstance(res, tuple) else res
        temp_firma = res[1] if isinstance(res, tuple) else st.session_state['firmalar']

        # MÜLAKAT SİMÜLASYONU
        final_ogr, final_firma, reddedilen_kisi = mulakat_simulasyonu(temp_ogr, temp_firma, red_orani)

        st.session_state['ogrenciler'] = final_ogr
        st.session_state['firmalar'] = final_firma

    except Exception as e:
        st.error(f"Annealing Hatası: {e}")
        st.stop()
    pb.empty()
    sure = time.time() - t1
    islem = True

# --- SONUÇLAR ---
if islem:
    # Yerleşenleri say (Yerleştiği_Firma sütunu None olmayanlar)
    yerlesen = st.session_state['ogrenciler']['Yerleştiği_Firma'].count()
    toplam = len(st.session_state['ogrenciler'])
    basari = (yerlesen / toplam) * 100 if toplam > 0 else 0
    puan = puan_hesapla(st.session_state['ogrenciler'])

    st.session_state['analiz_sonuclari'][algo] = basari

    st.success(f"✅ {algo} Tamamlandı!")

    if reddedilen_kisi > 0:
        st.warning(f"⚠️ Dikkat: Algoritma yerleştirdi ancak {reddedilen_kisi} öğrenci firma mülakatında elendi!")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Yerleşen", f"{yerlesen}/{toplam}")
    c2.metric("Başarı", f"%{basari:.1f}")
    c3.metric("Süre", f"{sure:.4f}s")
    c4.metric("Puan", f"{puan:,}".replace(",", "."))

# --- GÖRSELLEŞTİRME VE LİSTE ---
if btn_analiz:
    st.subheader("📊 Rapor")
    data = st.session_state['analiz_sonuclari']
    if data:
        c_g, c_t = st.columns([2, 1])
        with c_g:
            fig, ax = plt.subplots(figsize=(6, 3))
            bars = ax.bar(data.keys(), data.values(), color=['#FF4B4B', '#1C83E1', '#FFA500'])
            ax.set_ylim(0, 110)
            ax.set_ylabel("Başarı (%)")
            for b in bars: 
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 2, f"%{b.get_height():.1f}", ha='center')
            st.pyplot(fig)
        with c_t:
            st.dataframe(pd.DataFrame(list(data.items()), columns=['Algoritma', 'Başarı']), hide_index=True)
    else:
        st.info("Henüz analiz edilecek veri yok. Algoritmaları çalıştırın.")
else:
    st.subheader("📋 Liste")
    t1, t2 = st.tabs(["Öğrenciler", "Firmalar"])
    
    with t1:
        # --- HATA DÜZELTME: GÜVENLİ KOLON SEÇİMİ ---
        # Veri setindeki mevcut kolonları al
        df_ogr = st.session_state['ogrenciler']
        mevcut_kolonlar = df_ogr.columns.tolist()
        
        # Göstermek istediğimiz öncelikli kolonlar
        hedef_kolonlar = ['Öğrenci', 'GNO', 'Yerleştiği_Firma', 'Tercih1', 'Tercih2', 'Tercih3']
        
        # 'Ogrenci' vs 'Öğrenci' uyumsuzluğu varsa düzelt
        if 'Ogrenci' in mevcut_kolonlar and 'Öğrenci' not in mevcut_kolonlar:
            hedef_kolonlar = [k if k != 'Öğrenci' else 'Ogrenci' for k in hedef_kolonlar]
            
        # Sadece veri setinde GERÇEKTEN VAR OLAN kolonları seç (KeyError önler)
        gosterilecekler = [k for k in hedef_kolonlar if k in mevcut_kolonlar]
        
        # Eğer hedef kolonlardan hiçbiri yoksa, tüm tabloyu göster
        if not gosterilecekler:
             st.dataframe(df_ogr, use_container_width=True)
        else:
             st.dataframe(df_ogr[gosterilecekler], use_container_width=True)
             
    with t2:
        st.dataframe(st.session_state['firmalar'], use_container_width=True)