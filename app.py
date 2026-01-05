import streamlit as st
import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Stajyer Simülatörü", layout="wide", page_icon="🎓")

# --- CSS (Masaüstü Havası İçin) ---
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        font-weight: bold;
        border: 1px solid #ccc;
    }
    div[data-testid="stMetricValue"] {
        font-size: 26px;
        font-weight: bold;
    }
    .main-header {
        font-size: 32px;
        color: #1E3D59;
        text-align: center;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<div class="main-header">🎓 Stajyer Yerleştirme Simülasyonu</div>', unsafe_allow_html=True)

# --- İMPORTLAR ---
try:
    import veri_olustur
    import algo_greedy
    import algo_heuristic_hill_climbing
    import algo_heuristic_annealing
except ImportError as e:
    st.error(f"⚠️ Kritik Hata: Python dosyaları eksik! ({e})")
    st.stop()

# --- SESSION STATE ---
if 'ogrenciler' not in st.session_state: st.session_state['ogrenciler'] = pd.DataFrame()
if 'firmalar' not in st.session_state: st.session_state['firmalar'] = pd.DataFrame()
if 'analiz_sonuclari' not in st.session_state: st.session_state['analiz_sonuclari'] = {}

# --- YARDIMCI FONKSİYONLAR ---
def puan_hesapla(df):
    """Memnuniyet Puanı Hesabı (Masaüstüyle Birebir Aynı)"""
    if df.empty or 'Yerleştiği_Firma' not in df.columns: return 0
    puan_tablosu = {1: 100, 2: 85, 3: 70, 4: 50, 5: 30}
    toplam = 0
    for _, row in df[df['Yerleştiği_Firma'].notna()].iterrows():
        yf = row['Yerleştiği_Firma']
        for i in range(1, 6):
            col = f'Tercih{i}'
            if col in row and row[col] == yf:
                toplam += puan_tablosu.get(i, 10)
                break
    return toplam

def mulakat_simulasyonu(df_ogrenciler, df_firmalar, reddetme_orani):
    """
    Eğer oran 0 ise dokunmaz (Böylece 139 sonuç 139 kalır).
    """
    if reddetme_orani <= 0:
        return df_ogrenciler, df_firmalar, 0

    df_sonuc = df_ogrenciler.copy()
    reddedilen_sayisi = 0

    # Tutarlılık için seed'i burada sabitlemiyoruz, kaotik olsun diye
    # Ama ana veri üretiminde sabitledik.
    
    for idx, row in df_sonuc.iterrows():
        firma = row['Yerleştiği_Firma']
        if pd.notna(firma):
            zar = np.random.randint(0, 100)
            if zar < reddetme_orani:
                df_sonuc.at[idx, 'Yerleştiği_Firma'] = None
                
                # Kontenjanı iade et
                if 'Firma' in df_firmalar.columns:
                    f_idx = df_firmalar[df_firmalar['Firma'] == firma].index
                    if not f_idx.empty:
                        df_firmalar.at[f_idx[0], 'Kalan_Kontenjan'] += 1
                reddedilen_sayisi += 1

    return df_sonuc, df_firmalar, reddedilen_sayisi

# --- SIDEBAR (KONTROL PANELİ) ---
with st.sidebar:
    st.header("⚙️ Kontrol Paneli")
    
    st.subheader("1. Veri Ayarları")
    ogr_sayisi = st.number_input("Öğrenci Sayısı", 10, 5000, 150)
    firma_sayisi = st.number_input("Firma Sayısı", 5, 500, 40)
    
    # SEED AYARI (SONUÇLAR HEP AYNI ÇIKSIN DİYE)
    st.info("💡 'Sabit Veri' açıkken her basışta aynı öğrenciler oluşur (Masaüstü gibi).")
    sabit_veri = st.checkbox("Sabit Veri (Seed=42)", value=True)

    st.write("---")
    st.subheader("2. Zorluk Ayarları")
    # VARSAYILAN DEĞERİ 0 YAPTIM (Masaüstüyle aynı sonuç için)
    red_orani = st.slider("🚫 Mülakat Reddetme Oranı (%)", 0, 50, 0, 
                          help="0 yaparsan algoritma sonucu değişmez.")

    st.divider()

    if st.button("🎲 Veri Seti Oluştur", type="primary"):
        # Seed Sabitleme
        if sabit_veri:
            np.random.seed(42)
        else:
            np.random.seed(None)

        d1, d2 = veri_olustur.veri_seti_olustur(ogr_sayisi, firma_sayisi)

        # Dönüş sırasını kontrol et
        if 'Firma' in d1.columns:
            firmalar_df, ogrenciler_df = d1, d2
        else:
            ogrenciler_df, firmalar_df = d1, d2

        # İsimlendirme Düzeltme
        mapping = {'Ortalama': 'GNO', 'Not': 'GNO', 'Puan': 'GNO', 'gno': 'GNO', 
                   'Ogrenci_No': 'Öğrenci', 'Ogrenci': 'Öğrenci'}
        ogrenciler_df.rename(columns=mapping, inplace=True)
        
        if 'Yerleştiği_Firma' not in ogrenciler_df.columns:
            ogrenciler_df['Yerleştiği_Firma'] = None

        st.session_state['ogrenciler'] = ogrenciler_df
        st.session_state['firmalar'] = firmalar_df
        st.session_state['analiz_sonuclari'] = {} # Veri değişince analiz sıfırlanır
        
        st.success(f"✅ Veri Hazır: {len(ogrenciler_df)} Öğrenci")

    st.subheader("Algoritmalar")
    col_btns = st.columns(3)
    btn_greedy = col_btns[0].button("Greedy")
    btn_hill = col_btns[1].button("Hill Climb")
    btn_anneal = col_btns[2].button("Annealing")
    
    st.write("---")
    btn_kiyasla = st.button("📊 Simülasyon Analizi (Kıyaslama)", type="secondary")

    if st.button("🗑️ Sıfırla"):
        st.session_state.clear()
        st.rerun()

# --- GÜVENLİK KONTROLÜ ---
if st.session_state['ogrenciler'].empty:
    st.info("👈 Lütfen sol menüden **'Veri Seti Oluştur'** butonuna basın.")
    st.stop()

# --- ALGORİTMA ÇALIŞTIRMA ---
islem_bitti = False
secilen_algo = ""
islem_suresi = 0
red_sayisi = 0

if btn_greedy:
    secilen_algo = "Greedy"
    t_start = time.time()
    
    # Veriyi kopyalayarak gönder (Orijinal bozulmasın)
    res = algo_greedy.greedy_atama(st.session_state['ogrenciler'].copy(), st.session_state['firmalar'].copy())
    
    temp_ogr = res[0] if isinstance(res, tuple) else res
    temp_firma = st.session_state['firmalar'].copy() # Greedy firmayı değiştirmiyorsa
    
    # Mülakat (Red Oranı 0 ise çalışmaz)
    final_ogr, final_firma, red_sayisi = mulakat_simulasyonu(temp_ogr, temp_firma, red_orani)
    
    st.session_state['ogrenciler'] = final_ogr
    st.session_state['firmalar'] = final_firma
    islem_suresi = time.time() - t_start
    islem_bitti = True

elif btn_hill:
    secilen_algo = "Hill Climbing"
    t_start = time.time()
    bar = st.progress(0)
    
    def adim_guncelle(i):
        if i % 500 == 0: 
            bar.progress(min(i/3000, 1.0))
            time.sleep(0.0001)

    try:
        # Fonksiyon ismini bul
        if hasattr(algo_heuristic_hill_climbing, 'heuristic_atama'):
            func = algo_heuristic_hill_climbing.heuristic_atama
        elif hasattr(algo_heuristic_hill_climbing, 'hill_climbing_main'):
            func = algo_heuristic_hill_climbing.hill_climbing_main
        else:
            func = algo_heuristic_hill_climbing.hill_climbing
        
        res = func(st.session_state['ogrenciler'].copy(), st.session_state['firmalar'].copy(), 
                   iterasyon=3000, step_callback=adim_guncelle)
        
        temp_ogr = res[0] if isinstance(res, tuple) else res
        temp_firma = res[1] if isinstance(res, tuple) else st.session_state['firmalar']
        
        final_ogr, final_firma, red_sayisi = mulakat_simulasyonu(temp_ogr, temp_firma, red_orani)
        st.session_state['ogrenciler'] = final_ogr
        st.session_state['firmalar'] = final_firma
        
    except Exception as e:
        st.error(f"Hata: {e}")
        st.stop()
        
    bar.progress(100)
    time.sleep(0.2)
    bar.empty()
    islem_suresi = time.time() - t_start
    islem_bitti = True

elif btn_anneal:
    secilen_algo = "Simulated Annealing"
    t_start = time.time()
    bar = st.progress(0)
    
    def adim_guncelle(i):
        if i % 1000 == 0: 
            bar.progress(min(i/10000, 1.0))
            time.sleep(0.0001)
            
    try:
        if hasattr(algo_heuristic_annealing, 'heuristic_atama'):
            func = algo_heuristic_annealing.heuristic_atama
        elif hasattr(algo_heuristic_annealing, 'simulated_annealing_main'):
            func = algo_heuristic_annealing.simulated_annealing_main
        else:
            func = algo_heuristic_annealing.simulated_annealing
            
        res = func(st.session_state['ogrenciler'].copy(), st.session_state['firmalar'].copy(), 
                   iterasyon=10000, step_callback=adim_guncelle)
        
        temp_ogr = res[0] if isinstance(res, tuple) else res
        temp_firma = res[1] if isinstance(res, tuple) else st.session_state['firmalar']
        
        final_ogr, final_firma, red_sayisi = mulakat_simulasyonu(temp_ogr, temp_firma, red_orani)
        st.session_state['ogrenciler'] = final_ogr
        st.session_state['firmalar'] = final_firma
        
    except Exception as e:
        st.error(f"Hata: {e}")
        st.stop()
        
    bar.progress(100)
    time.sleep(0.2)
    bar.empty()
    islem_suresi = time.time() - t_start
    islem_bitti = True

# --- SONUÇLARI KAYDET VE GÖSTER ---
if islem_bitti:
    df_son = st.session_state['ogrenciler']
    yerlesen_sayisi = df_son['Yerleştiği_Firma'].count()
    toplam_ogr = len(df_son)
    basari_orani = (yerlesen_sayisi / toplam_ogr) * 100
    toplam_puan = puan_hesapla(df_son)
    
    # Analiz geçmişine kaydet
    st.session_state['analiz_sonuclari'][secilen_algo] = {
        "Yerleşen": yerlesen_sayisi,
        "Başarı (%)": round(basari_orani, 2),
        "Puan": toplam_puan,
        "Süre (sn)": round(islem_suresi, 4)
    }

    st.success(f"✅ **{secilen_algo} Tamamlandı!**")
    
    if red_sayisi > 0:
        st.warning(f"⚠️ Mülakat Sonucu: {red_sayisi} kişi algoritma yerleştirmesine rağmen elendi!")
    elif red_orani == 0:
        st.info("ℹ️ Mülakat Reddetme kapalı olduğu için saf algoritma sonucu gösteriliyor.")

    # Metrikler
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Yerleşen", f"{yerlesen_sayisi} / {toplam_ogr}", delta_color="normal")
    m2.metric("Başarı Oranı", f"%{basari_orani:.1f}")
    m3.metric("Memnuniyet Puanı", f"{toplam_puan:,}".replace(",", "."))
    m4.metric("İşlem Süresi", f"{islem_suresi:.3f} sn")

# --- SİMÜLASYON ANALİZİ (KIYASLAMA EKRANI) ---
if btn_kiyasla:
    st.divider()
    st.subheader("📊 Simülasyon Analizi ve Algoritma Karşılaştırması")
    
    sonuclar = st.session_state['analiz_sonuclari']
    
    if not sonuclar:
        st.warning("⚠️ Henüz hiçbir algoritma çalıştırılmadı. Kıyaslama yapmak için yukarıdan algoritmaları sırayla çalıştırın.")
    else:
        # Tablo Haline Getir
        df_analiz = pd.DataFrame(sonuclar).T.reset_index().rename(columns={"index": "Algoritma"})
        
        col_tablo, col_grafik = st.columns([1, 1])
        
        with col_tablo:
            st.markdown("##### 📋 Sayısal Veriler")
            st.dataframe(df_analiz, use_container_width=True, hide_index=True)
            
            # En iyiyi bul
            best_algo = df_analiz.loc[df_analiz['Puan'].idxmax()]
            st.success(f"🏆 **Kazanan:** {best_algo['Algoritma']} (Puan: {int(best_algo['Puan'])})")

        with col_grafik:
            st.markdown("##### 📈 Memnuniyet Puanı Karşılaştırması")
            # Basit Matplotlib Grafiği
            fig, ax = plt.subplots(figsize=(5, 3))
            colors = ['#FF4B4B', '#1C83E1', '#FFA500', '#00CC96']
            # Renkleri sırayla ata
            bar_colors = [colors[i % len(colors)] for i in range(len(df_analiz))]
            
            bars = ax.bar(df_analiz['Algoritma'], df_analiz['Puan'], color=bar_colors)
            ax.set_ylabel("Puan")
            ax.set_title("Algoritma Performansı")
            
            # Üstüne değerleri yaz
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}',
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            st.pyplot(fig)

# --- LİSTE GÖRÜNÜMÜ ---
st.divider()
st.subheader("📋 Detaylı Listeler")
tab1, tab2 = st.tabs(["👨‍🎓 Öğrenci Listesi", "🏢 Firma Listesi"])

with tab1:
    df_ogr = st.session_state['ogrenciler']
    cols = df_ogr.columns.tolist()
    
    # Görüntülenecek sütunları temizle
    ideal_cols = ['Öğrenci', 'GNO', 'Yerleştiği_Firma', 'Tercih1', 'Tercih2']
    if 'Ogrenci' in cols: ideal_cols = [c if c != 'Öğrenci' else 'Ogrenci' for c in ideal_cols]
    final_cols = [c for c in ideal_cols if c in cols]
    
    if final_cols:
        st.dataframe(df_ogr[final_cols], use_container_width=True)
    else:
        st.dataframe(df_ogr, use_container_width=True)

with tab2:
    st.dataframe(st.session_state['firmalar'], use_container_width=True)