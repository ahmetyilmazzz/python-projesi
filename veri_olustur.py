import pandas as pd
import numpy as np

def veri_seti_olustur(ogrenci_sayisi, firma_sayisi, tercih_sayisi=5, seed=46):
    if firma_sayisi > ogrenci_sayisi:
        raise ValueError("firma_sayisi, ogrenci_sayisi'ndan büyük olamaz.")
    if firma_sayisi < tercih_sayisi:
        raise ValueError("firma_sayisi, tercih_sayisi'ndan küçük olamaz.")

    rng = np.random.default_rng(seed)

    firma_isimleri = [f"Firma_{i + 1}" for i in range(firma_sayisi)]

    # 1. ADIM: Her firmaya en az 1 kontenjan ver
    kontenjanlar = np.ones(firma_sayisi)   # tipi numpy.ndarray

    # 2. ADIM: Geriye kalan (Öğrenci - Firma) sayısını rastgele dağıt
    # Matematiksel olarak: FirmaSayisi * 1 + (Ogrenci - FirmaSayisi) = OgrenciSayisi
    # Yani toplam tam olarak eşitlenir.
    kalan = ogrenci_sayisi - firma_sayisi


    # aşağıdaki koşul kontenjan sayısının, öğrenci sayısına eşit olduğunun kanıtı
    if kalan > 0:
        # secimler kalan öğrenci sayısı uzunluğunda bir ndarray tutar ve kalan kontenjanları random yerleştirir.
        secimler = rng.integers(0, firma_sayisi, size=kalan)  # numpy ndarray

        # listede hangi sayıdan kaç tane olduğunu sayar ve sırayla yazar
        kontenjanlar += np.bincount(secimler, minlength=firma_sayisi) # minlength: hiç seçilmeyen olsa bile onlara sıfır koy.

    # firmalar dataframe i
    df_firmalar = pd.DataFrame({
        "Firma": firma_isimleri,
        "Kontenjan": kontenjanlar
    })

    # güzel görünsün diye dataframe e sonradan ekledim
    df_firmalar["Kalan_Kontenjan"] = df_firmalar["Kontenjan"]

    # öğrencileri oluşturma snippetı
    ogrenciler = []
    for i in range(ogrenci_sayisi):
        isim = f"Ogrenci_{i + 1}"
        gno = np.round(rng.uniform(2.0, 4.0), 2)

        #öğrenci tercihlerini yapar firma isimleri listesinden 5 tane seçer ve seçilen tekrar seçilemesin
        tercihler = rng.choice(firma_isimleri, size=tercih_sayisi, replace=False)
        ogrenciler.append([isim, gno, *tercihler])  # * işareti paketi açmak için yani liste içinde liste olmasın

    df_ogrenciler = pd.DataFrame(
        ogrenciler,
        columns=["Öğrenci", "GNO"] + [f"Tercih{j}" for j in range(1, tercih_sayisi + 1)]
    )

    df_firmalar.to_csv("proje_firmalar.csv", index=False, encoding="utf-8-sig")
    df_ogrenciler.to_csv("proje_ogrenciler.csv", index=False, encoding="utf-8-sig")

    print(
        "Oluşturuldu: proje_firmalar.csv ve proje_ogrenciler.csv\n"
        + "Toplam kontenjan: "
        + str(int(df_firmalar["Kontenjan"].sum()))
        + " (Öğrenci Sayısı: " + str(ogrenci_sayisi) + ")"
    )
    return df_firmalar, df_ogrenciler


if __name__ == "__main__":
    veri_seti_olustur(ogrenci_sayisi=150, firma_sayisi=40)
