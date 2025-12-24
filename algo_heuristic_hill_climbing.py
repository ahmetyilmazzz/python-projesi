import pandas as pd
import numpy as np


def memnuniyet_skoru_hesapla(ogrenciler_df):
    puanlar = {1: 100, 2: 80, 3: 60, 4: 40, 5: 20}
    toplam_puan = 0
    yerlesenler = ogrenciler_df[ogrenciler_df['Yerleştiği_Firma'].notna()]

    for _, ogrenci in yerlesenler.iterrows():
        yerlesen_firma = ogrenci['Yerleştiği_Firma']
        for i in range(1, 6):
            if ogrenci[f'Tercih{i}'] == yerlesen_firma:
                toplam_puan += puanlar[i]
                break
    return toplam_puan


def heuristic_atama(ogrenciler_df, firmalar_df, iterasyon=3000, step_callback=None):
    best_ogrenciler = ogrenciler_df.copy().reset_index(drop=True)
    best_firmalar = firmalar_df.copy().reset_index(drop=True)

    if 'Yerlesenler' not in best_firmalar.columns:
        best_firmalar['Yerlesenler'] = None
    if 'Tercih_Sırası' not in best_ogrenciler.columns:
        best_ogrenciler['Tercih_Sırası'] = "-"

    best_score = memnuniyet_skoru_hesapla(best_ogrenciler)
    ogrenci_indexleri = best_ogrenciler.index.tolist()

    for i in range(iterasyon):
        if step_callback and i % 50 == 0:
            step_callback(i)

        secilen_idx = np.random.choice(ogrenci_indexleri)
        temp_ogr = best_ogrenciler.copy()
        temp_frm = best_firmalar.copy()

        secilen_ogr = temp_ogr.loc[secilen_idx]
        mevcut_firma = secilen_ogr['Yerleştiği_Firma']

        tercih_no = np.random.randint(1, 6)
        hedef_firma = secilen_ogr[f'Tercih{tercih_no}']

        if mevcut_firma == hedef_firma: continue

        f_row = temp_frm[temp_frm['Firma'] == hedef_firma]
        if f_row.empty: continue
        f_idx = f_row.index[0]

        degisim_yapildi = False

        if temp_frm.at[f_idx, 'Kontenjan'] > 0:
            temp_ogr.at[secilen_idx, 'Yerleştiği_Firma'] = hedef_firma
            temp_frm.at[f_idx, 'Kontenjan'] -= 1

            if pd.notna(mevcut_firma):
                eski_f_row = temp_frm[temp_frm['Firma'] == mevcut_firma]
                if not eski_f_row.empty:
                    temp_frm.at[eski_f_row.index[0], 'Kontenjan'] += 1
            degisim_yapildi = True

        else:
            ordaki_ogrenciler = temp_ogr[temp_ogr['Yerleştiği_Firma'] == hedef_firma]
            if not ordaki_ogrenciler.empty:
                takas_idx = np.random.choice(ordaki_ogrenciler.index)
                temp_ogr.at[secilen_idx, 'Yerleştiği_Firma'] = hedef_firma
                temp_ogr.at[takas_idx, 'Yerleştiği_Firma'] = mevcut_firma
                degisim_yapildi = True

        if degisim_yapildi:
            yeni_skor = memnuniyet_skoru_hesapla(temp_ogr)
            if yeni_skor > best_score:
                best_score = yeni_skor
                best_ogrenciler = temp_ogr
                best_firmalar = temp_frm

    # --- FİNAL GÜNCELLEMELERİ ---

    # 1. Firmalara yerleşenleri yaz
    yerlesen_grup = best_ogrenciler[best_ogrenciler['Yerleştiği_Firma'].notna()].groupby('Yerleştiği_Firma')[
        'Öğrenci'].apply(lambda x: ", ".join(x))
    best_firmalar['Yerlesenler'] = best_firmalar['Firma'].map(yerlesen_grup).fillna("-")

    # 2. Öğrencilerin "Tercih Sırası"nı (1., 2. vs) tekrar hesapla
    # Çünkü heuristic yerleri değiştirdi, eski sıra bilgisi yanlış olabilir.
    for idx, row in best_ogrenciler.iterrows():
        firma = row['Yerleştiği_Firma']
        if pd.isna(firma):
            best_ogrenciler.at[idx, 'Tercih_Sırası'] = "-"
        else:
            # Hangi tercih olduğunu bul
            for k in range(1, 6):
                if row[f'Tercih{k}'] == firma:
                    best_ogrenciler.at[idx, 'Tercih_Sırası'] = k
                    break

    return best_ogrenciler, best_firmalar