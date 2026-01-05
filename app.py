import streamlit as st
import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Stajyer Simülatörü", layout="wide", page_icon="🎓")

# --- CSS (Butonlar Daha Şık Olsun) ---
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        font-weight: bold;
        border: 1px solid #ddd;
    }
    div[data-testid="stMetricValue"] {
        font-size: 24px;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🎓 Stajyer Yerleştirme Simülasyonu")

# --- İMPORTLAR ---
try:
    import veri_olustur
    import algo_greedy
    import algo_heuristic_hill_climbing
    import algo_heuristic_annealing
except ImportError as e:
    st.error(f"Dosyalar eksik: {e}")
    st.stop()

# --- SESSION STATE ---
if 'ogrenciler' not in st.session_state: st.session_state['ogrenciler'] = pd.DataFrame()
if 'firmalar' not in st.session_state: st.session_state['firmalar'] = pd.DataFrame()
if 'analiz_sonuclari' not in st.session_state: st.session_state['analiz_sonuclari'] = {}

# --- YARDIMCI FONKSİYONLAR ---
def puan_hesapla(df):
    if df.empty or 'Yerleştiği_Firma' not in df.columns: return 0
    puan_tablosu = {1: 100, 2: 85, 3: 70, 4: 50, 5: 30}
    toplam = 0
    for _, row in df[df['Yerleştiği_Firma'].notna()].iterrows():
        yf = row['Yerleştiği_Firma']
        for i in range(1, 6):
            if f'Tercih{i}' in row and row[f'Tercih{i}'] == yf:
                toplam += puan_tablosu.get(i, 10)
                break
    return toplam

# --- SIDEBAR (SADELEŞTİRİLMİŞ) ---
with st.sidebar:
    st.header("⚙️ Ayarlar")
    
    # Sadece Gerekli Girdiler
    ogr_sayisi = st.number_input("Öğrenci Sayısı", value=150)
    firma_sayisi = st.number_input("Firma Sayısı", value=40)
    
    # Veri Oluştur Butonu (Vurgulu)
    if st.button("🎲 Veri Oluştur", type="primary"):
        np.random.seed(42) # Sabit sonuç için gizli ayar
        
        d1, d2 = veri_olustur.veri_seti_olustur(ogr_sayisi, firma_sayisi)
        
        # Hangi df hangisi kontrolü
        if 'Firma' in d1.columns:
            firmalar_df, ogrenciler_df = d1, d2
        else:
            ogrenciler_df, firmalar_df = d1, d2

        # İsim Düzeltme
        mapping = {'Ortalama': 'GNO', 'Not': 'GNO', 'Puan': 'GNO', 'Ogrenci_No': 'Öğrenci'}
        ogrenciler_df.rename(columns=mapping, inplace=True)
        if 'Yerleştiği_Firma' not in ogrenciler_df.columns:
            ogrenciler_df['Yerleştiği_Firma'] = None

        st.session_state['ogrenciler'] = ogrenciler_df
        st.session_state['firmalar'] = firmalar_df
        st.session_state['analiz_sonuclari'] = {}
        st.success("Veri Hazır.")

    st.markdown("---")
    st.subheader("Algoritmalar")
    
    # Alt alta butonlar
    btn_greedy = st.button("🚀 Greedy")
    btn_hill = st.button("⛰️ Hill Climbing")
    btn_anneal = st.button("🔥 Annealing")
    
    st.markdown("---")
    btn_kiyasla = st.button("📊 Analiz & Kıyasla")
    
    if st.button("🗑️ Sıfırla"):
        st.session_state.clear()
        st.rerun()

# --- ANA EKRAN MANTIĞI ---
if st.session_state['ogrenciler'].empty:
    st.info("👈 Sol menüden 'Veri Oluştur' butonuna basın.")
    st.stop()

islem_bitti = False
secilen_algo = ""
sure = 0

# ALGORITMA ÇALIŞTIRMA (Reddetme Oranı = 0 Gizli)
if btn_greedy:
    secilen_algo = "Greedy"
    t1 = time.time()
    res = algo_greedy.greedy_atama(st.session_state['ogrenciler'].copy(), st.session_state['firmalar'].copy())
    
    # Sonuç işleme
    temp_ogr = res[0] if isinstance(res, tuple) else res
    st.session_state['ogrenciler'] = temp_ogr
    if isinstance(res, tuple): st.session_state['firmalar'] = res[1]
    
    sure = time.time() - t1
    islem_bitti = True

elif btn_hill:
    secilen_algo = "Hill Climbing"
    t1 = time.time()
    bar = st.progress(0)
    
    def step(i): 
        if i % 500 == 0: bar.progress(min(i/3000, 1.0))
    
    try:
        # Doğru fonksiyonu bul
        if hasattr(algo_heuristic_hill_climbing, 'hill_climbing_main'):
            func = algo_heuristic_hill_climbing.hill_climbing_main
        else:
            func = algo_heuristic_hill_climbing.hill_climbing
            
        res = func(st.session_state['ogrenciler'].copy(), st.session_state['firmalar'].copy(), iterasyon=3000, step_callback=step)
        
        st.session_state['ogrenciler'] = res[0] if isinstance(res, tuple) else res
        if isinstance(res, tuple): st.session_state['firmalar'] = res[1]
        
    except Exception as e: st.error(e)
    
    bar.empty()
    sure = time.time() - t1
    islem_bitti = True

elif btn_anneal:
    secilen_algo = "Simulated Annealing"
    t1 = time.time()
    bar = st.progress(0)
    
    def step(i):
        if i % 1000 == 0: bar.progress(min(i/10000, 1.0))
        
    try:
        if hasattr(algo_heuristic_annealing, 'simulated_annealing_main'):
            func = algo_heuristic_annealing.simulated_annealing_main
        else:
            func = algo_heuristic_annealing.simulated_annealing
            
        res = func(st.session_state['ogrenciler'].copy(), st.session_state['firmalar'].copy(), iterasyon=10000, step_callback=step)
        
        st.session_state['ogrenciler'] = res[0] if isinstance(res, tuple) else res
        if isinstance(res, tuple): st.session_state['firmalar'] = res[1]
        
    except Exception as e: st.error(e)
    
    bar.empty()
    sure = time.time() - t1
    islem_bitti = True

# --- SONUÇLARI GÖSTER ---
if islem_bitti:
    df = st.session_state['ogrenciler']
    yerlesen = df['Yerleştiği_Firma'].count()
    basari = (yerlesen / len(df)) * 100
    puan = puan_hesapla(df)
    
    st.session_state['analiz_sonuclari'][secilen_algo] = {"Puan": puan, "Yerleşen": yerlesen}
    
    st.success(f"✅ {secilen_algo} Tamamlandı")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Yerleşen", f"{yerlesen}/{len(df)}")
    c2.metric("Başarı", f"%{basari:.1f}")
    c3.metric("Puan", f"{puan}")
    c4.metric("Süre", f"{sure:.3f}s")

# --- KIYASLAMA ---
if btn_kiyasla:
    st.subheader("📊 Analiz")
    res = st.session_state['analiz_sonuclari']
    if res:
        df_res = pd.DataFrame(res).T.reset_index().rename(columns={'index':'Algo'})
        c1, c2 = st.columns(2)
        c1.dataframe(df_res, hide_index=True, use_container_width=True)
        
        fig, ax = plt.subplots(figsize=(5,3))
        ax.bar(df_res['Algo'], df_res['Puan'], color=['#FF4B4B','#1C83E1','#FFA500'])
        c2.pyplot(fig)
    else:
        st.warning("Önce algoritmaları çalıştırın.")

# --- LİSTE ---
st.divider()
t1, t2 = st.tabs(["Öğrenciler", "Firmalar"])
with t1:
    df = st.session_state['ogrenciler']
    # Sadece var olan kolonları göster
    cols = ['Öğrenci', 'GNO', 'Yerleştiği_Firma', 'Tercih1', 'Tercih2']
    cols = [c if c != 'Öğrenci' and 'Ogrenci' in df.columns else c for c in cols] # isim düzeltme
    valid_cols = [c for c in cols if c in df.columns]
    st.dataframe(df[valid_cols] if valid_cols else df, use_container_width=True)
with t2:
    st.dataframe(st.session_state['firmalar'], use_container_width=True)