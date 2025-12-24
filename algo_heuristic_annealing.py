import pandas as pd
import numpy as np


def memnuniyet_skoru_hesapla(ogrenciler_df):
    """
    Öğrencilerin yerleştiği firmaya göre puanını hesaplar.
    """
    puanlar = {1: 100, 2: 80, 3: 60, 4: 40, 5: 20}
    toplam_puan = 0

    # Yerleşenleri filtrele (Hata almamak için kopyasıyla çalış)
    df = ogrenciler_df.copy()
    if 'Yerleştiği_Firma' not in df.columns:
        return 0

    yerlesenler = df[df['Yerleştiği_Firma'].notna()]

    for _, ogrenci in yerlesenler.iterrows():
        yerlesen_firma = ogrenci['Yerleştiği_Firma']
        # 1'den 5'e kadar tercihleri kontrol et
        for i in range(1, 6):
            col_name = f'Tercih{i}'
            # Eğer tercih sütunu varsa ve eşleşiyorsa
            if col_name in ogrenci and ogrenci[col_name] == yerlesen_firma:
                toplam_puan += puanlar[i]
                break
    return toplam_puan


def heuristic_atama(ogrenciler_df, firmalar_df, iterasyon=10000, step_callback=None):
    """
    Simulated Annealing - Hassas Ayar Modu (Greedy üzerine iyileştirme)
    """
    # 1. Verileri Güvenli Kopyala
    current_ogrenciler = ogrenciler_df.copy().reset_index(drop=True)
    current_firmalar = firmalar_df.copy().reset_index(drop=True)

    # Kolon yoksa oluştur
    if 'Yerleştiği_Firma' not in current_ogrenciler.columns:
        current_ogrenciler['Yerleştiği_Firma'] = None

    # Başlangıç Skoru (Genelde Greedy'den gelir)
    current_score = memnuniyet_skoru_hesapla(current_ogrenciler)

    # EN İYİ Çözümü Sakla
    best_ogrenciler = current_ogrenciler.copy()
    best_firmalar = current_firmalar.copy()
    best_score = current_score

    # --- AYARLAR (GÜNCELLENDİ) ---
    # Önceden 5000 idi, şimdi 150 yapıyoruz.
    # Bu sayede eldeki güzel çözümü bozmadan ufak iyileştirmeler arayacak.
    sicaklik = 150
    soguma_orani = 0.99

    ogrenci_indexleri = current_ogrenciler.index.tolist()

    print(f"--- HEURISTIC BAŞLIYOR (Fine-Tuning Modu) ---")
    print(f"Başlangıç Skoru: {current_score}")

    for i in range(iterasyon):
        # Arayüze sinyal gönder (Progress Bar)
        if step_callback and i % 50 == 0:
            step_callback(i)

        # --- 1. KOMŞU ÇÖZÜM ÜRET (HAMLE) ---
        # Rastgele bir öğrenci seç
        secilen_idx = np.random.choice(ogrenci_indexleri)
        eski_firma = current_ogrenciler.at[secilen_idx, 'Yerleştiği_Firma']

        # Rastgele bir tercihine gitmeye çalış (1..5)
        tercih_no = np.random.randint(1, 6)
        col_name = f'Tercih{tercih_no}'

        if col_name not in current_ogrenciler.columns: continue
        hedef_firma = current_ogrenciler.at[secilen_idx, col_name]

        # Boş tercih, aynı yer veya tercih yoksa pas geç
        if pd.isna(hedef_firma) or eski_firma == hedef_firma:
            # Boşa dönmesin, soğutmayı hafiflet
            continue

        # Hedef firmanın verisine ulaş
        f_rows = current_firmalar[current_firmalar['Firma'] == hedef_firma]
        if f_rows.empty: continue
        f_idx = f_rows.index[0]

        islem_tipi = None
        takas_edilen_idx = None

        # A) Kontenjan Var -> MOVE
        if current_firmalar.at[f_idx, 'Kontenjan'] > 0:
            islem_tipi = "MOVE"
            current_ogrenciler.at[secilen_idx, 'Yerleştiği_Firma'] = hedef_firma
            current_firmalar.at[f_idx, 'Kontenjan'] -= 1

            # Eski yerinden düş (Eğer bir yerde çalışıyorduysa)
            if pd.notna(eski_firma):
                eski_rows = current_firmalar[current_firmalar['Firma'] == eski_firma]
                if not eski_rows.empty:
                    current_firmalar.at[eski_rows.index[0], 'Kontenjan'] += 1

        # B) Kontenjan Yok -> SWAP (Takas)
        else:
            islem_tipi = "SWAP"
            # O firmadaki öğrencilerden birini seç
            ordaki_ogrenciler = current_ogrenciler[current_ogrenciler['Yerleştiği_Firma'] == hedef_firma]
            if ordaki_ogrenciler.empty: continue

            takas_edilen_idx = np.random.choice(ordaki_ogrenciler.index)

            # Değiştir
            current_ogrenciler.at[secilen_idx, 'Yerleştiği_Firma'] = hedef_firma
            current_ogrenciler.at[takas_edilen_idx, 'Yerleştiği_Firma'] = eski_firma

        # --- 2. DEĞERLENDİRME ---
        yeni_score = memnuniyet_skoru_hesapla(current_ogrenciler)
        delta = yeni_score - current_score

        kabul_edildi = False

        if delta > 0:
            kabul_edildi = True
        else:
            # Delta negatifse (skor düşüyorsa)
            # Sıcaklık düşük olduğu için, büyük düşüşleri kabul etmeyecek
            if sicaklik > 0.001:
                olasilik = np.exp(delta / sicaklik)
                if np.random.rand() < olasilik:
                    kabul_edildi = True

        # --- 3. SONUÇ ---
        if kabul_edildi:
            current_score = yeni_score
            if current_score > best_score:
                best_score = current_score
                best_ogrenciler = current_ogrenciler.copy()
                best_firmalar = current_firmalar.copy()
                print(f"!!! GELİŞME VAR: {best_score} (+{best_score - 12660} puan) (Iterasyon {i})")
        else:
            # GERİ AL (UNDO)
            if islem_tipi == "MOVE":
                current_ogrenciler.at[secilen_idx, 'Yerleştiği_Firma'] = eski_firma
                current_firmalar.at[f_idx, 'Kontenjan'] += 1
                if pd.notna(eski_firma):
                    eski_rows = current_firmalar[current_firmalar['Firma'] == eski_firma]
                    if not eski_rows.empty:
                        current_firmalar.at[eski_rows.index[0], 'Kontenjan'] -= 1

            elif islem_tipi == "SWAP":
                current_ogrenciler.at[secilen_idx, 'Yerleştiği_Firma'] = eski_firma
                current_ogrenciler.at[takas_edilen_idx, 'Yerleştiği_Firma'] = hedef_firma

        # Soğutma
        sicaklik *= soguma_orani

        # Loglama (Sıklığı azalttım)
        if i % 1000 == 0:
            print(f"Iterasyon {i}: Skor {current_score}, Best {best_score}, T={sicaklik:.2f}")

    print(f"--- HEURISTIC BİTTİ. Final Skor: {best_score} ---")

    # --- RAPORLAMA HAZIRLIĞI ---
    # Yerleşenleri yaz
    best_firmalar['Yerlesenler'] = None
    yerlesen_grup = best_ogrenciler[best_ogrenciler['Yerleştiği_Firma'].notna()].groupby('Yerleştiği_Firma')[
        'Öğrenci'].apply(lambda x: ", ".join(str(s) for s in x))
    best_firmalar['Yerlesenler'] = best_firmalar['Firma'].map(yerlesen_grup).fillna("-")

    # Sıralamayı güncelle
    best_ogrenciler['Tercih_Sırası'] = "-"
    for idx, row in best_ogrenciler.iterrows():
        firma = row['Yerleştiği_Firma']
        if pd.notna(firma):
            for k in range(1, 6):
                col = f'Tercih{k}'
                if col in row and row[col] == firma:
                    best_ogrenciler.at[idx, 'Tercih_Sırası'] = k
                    break

    return best_ogrenciler, best_firmalar