import pandas as pd
import numpy as np


def greedy_atama(ogrenciler_df, firmalar_df):
    ogrenciler = ogrenciler_df.copy().reset_index(drop=True)
    firmalar = firmalar_df.copy().reset_index(drop=True)

    # Kolonları temizle/başlat
    ogrenciler['Yerleştiği_Firma'] = None
    ogrenciler['Tercih_Sırası'] = "-"
    firmalar['Yerlesenler'] = None

    # GNO'ya göre sırala (En yüksek puanlı en önce seçer)
    sirali_ogrenciler = ogrenciler.sort_values('GNO', ascending=False)


    for idx, ogrenci in sirali_ogrenciler.iterrows():  # enumerate ile aynı işi yapıyor 
        # 1'den 5'e kadar tercihleri dön
        for i in range(1, 6):
            tercih_col = f'Tercih{i}'

            if tercih_col not in ogrenci:
                continue

            # öğrencinin ilk tercihini alıyor
            tercih_firma = ogrenci[tercih_col]

            # boolean mask yapıp o firmanın satırını getiriyor
            firma_row = firmalar[firmalar['Firma'] == tercih_firma]
            if firma_row.empty:
                continue

            firma_idx = firma_row.index[0]

            # Kontenjan varsa yerleştir
            if firmalar.at[firma_idx, 'Kontenjan'] > 0:
                ogrenciler.at[idx, 'Yerleştiği_Firma'] = tercih_firma
                ogrenciler.at[idx, 'Tercih_Sırası'] = i
                firmalar.at[firma_idx, 'Kontenjan'] -= 1
                break

    # firmalara yerleşen öğrencileri gruplar firma dataframeinde Yerlesenler sütununa yerleşenleri ekler
    yerlesen_grup = ogrenciler[ogrenciler['Yerleştiği_Firma'].notna()].groupby('Yerleştiği_Firma')['Öğrenci'].apply(
        lambda x: ", ".join(x))
    firmalar['Yerlesenler'] = firmalar['Firma'].map(yerlesen_grup).fillna("-")

    return ogrenciler, firmalar

ogrenciler1 = pd.read_csv("proje_ogrenciler.csv")
firmalar1 = pd.read_csv("proje_firmalar.csv")
# greedy_atama(ogrenciler1, firmalar1)

def simulasyon_dongusu(ogrenciler_df, firmalar_df):
    """
    Simülasyon Mantığı:
    1. Greedy ile yerleştir.
    2. Firmalar "red_olasiligi" oranında öğrenciyi kovar.
    3. red_olasiligi her turda azalır (Sistem oturmaya başlar).
    4. Kontenjan dolana veya hareket bitene kadar devam eder.
    """
    ogr = ogrenciler_df.copy().reset_index(drop=True)
    frm = firmalar_df.copy().reset_index(drop=True)

    # Temizlik
    if 'Yerleştiği_Firma' not in ogr.columns:
        ogr['Yerleştiği_Firma'] = None
    ogr['Tercih_Sırası'] = "-"

    gecmis_log = []
    max_tur = 10

    # Başlangıç Red Olasılığı (%15 ile başla, giderek düşür)
    base_red_olasiligi = 0.15

    for tur in range(1, max_tur + 1):

        # boştakileri yerleştir
        bostakiler = ogr[ogr['Yerleştiği_Firma'].isnull()].sort_values('GNO', ascending=False)
        yerlesen_bu_tur = 0

        for idx, ogrenci in bostakiler.iterrows():
            for i in range(1, 6):
                tercih = ogrenci[f'Tercih{i}']
                f_idx = frm[frm['Firma'] == tercih].index
                # print(f_idx)
                if not f_idx.empty and frm.at[f_idx[0], 'Kontenjan'] > 0:
                    ogr.at[idx, 'Yerleştiği_Firma'] = tercih
                    ogr.at[idx, 'Tercih_Sırası'] = i
                    frm.at[f_idx[0], 'Kontenjan'] -= 1
                    yerlesen_bu_tur += 1
                    break

        # --- ADIM 2: RED SİMÜLASYONU (DİNAMİK ORAN) ---
        # Her turda red olasılığını azaltıyoruz ki sistem otursun (Converge etsin)
        # Örnek: Tur 1 -> %15, Tur 2 -> %12 ... Tur 5 -> %0 (Artık kimse atılmasın)
        guncel_red_orani = max(0, base_red_olasiligi - (tur * 0.03))

        reddedilen_sayisi = 0
        red_listesi_text = []

        # Sadece bu tur yerleşenler değil, halihazırda çalışanlar da risk altında olsun
        calisanlar = ogr[ogr['Yerleştiği_Firma'].notna()]

        if not calisanlar.empty and guncel_red_orani > 0:
            for idx, row in calisanlar.iterrows():
                # Her öğrenci için zar at
                if np.random.rand() < guncel_red_orani:
                    firma = row['Yerleştiği_Firma']
                    ogr.at[idx, 'Yerleştiği_Firma'] = None
                    ogr.at[idx, 'Tercih_Sırası'] = "-"  # Sırası silinir

                    # Firmanın kontenjanını geri ver
                    f_idx = frm[frm['Firma'] == firma].index[0]
                    frm.at[f_idx, 'Kontenjan'] += 1

                    reddedilen_sayisi += 1
                    red_listesi_text.append(f"{firma}->{row['Öğrenci']}")

        # --- LOGLAMA ---
        kalan_kontenjan = frm['Kontenjan'].sum()

        gecmis_log.append({
            "Tur": tur,
            "Yerleşen": yerlesen_bu_tur,
            "Reddedilen": reddedilen_sayisi,
            "Kalan_Kontenjan": kalan_kontenjan,
            "Red_Detay": ", ".join(red_listesi_text[:3]) + ("..." if len(red_listesi_text) > 3 else "")
        })

        # --- ÇIKIŞ KOŞULU ---
        # Eğer kimse yerleşmediyse VE kimse reddedilmediyse döngü bitmiştir.
        if yerlesen_bu_tur == 0 and reddedilen_sayisi == 0:
            break

        # Eğer kontenjan tamamen dolduysa bitir (Proje isterindeki "Tüm öğrenciler yerleşene kadar" mantığına yakın)
        if kalan_kontenjan == 0 and reddedilen_sayisi == 0:
            break

    return ogr, frm, gecmis_log

simulasyon_dongusu(ogrenciler1, firmalar1)