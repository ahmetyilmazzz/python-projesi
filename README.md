# 🎯 Stajyer Yerleştirme Simülasyonu (Intern Placement Simulation)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ahmet-yilmaz--intern-placement-simulation.streamlit.app)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

> **NP-Hard** sınıfındaki atama problemlerine yönelik; Greedy, Hill Climbing ve Simulated Annealing algoritmalarını kullanan hibrit bir **Karar Destek Sistemi.**

## 📖 Proje Hakkında

Bu proje, üniversite öğrencilerinin şirketlere stajyer olarak yerleştirilmesi sürecini optimize etmek amacıyla geliştirilmiştir. Manuel atamaların yarattığı verimsizliği ve adaletsizliği ortadan kaldırmak için **Sezgisel (Heuristic) Optimizasyon Algoritmaları** kullanır.

Sistem, öğrencilerin **Genel Not Ortalaması (GNO)** ve **Tercih Sıralamalarını** baz alarak; toplam **Memnuniyet Skorunu (Global Optimum)** maksimize etmeye çalışır. Ayrıca gerçek hayat senaryolarını simüle etmek için "Mülakat/Reddedilme" gibi stokastik parametreler içerir.

### 🌟 Temel Özellikler
* **Çift Arayüz Desteği:** * 🖥️ **Masaüstü:** PyQt5 ile geliştirilmiş, detaylı yönetim paneli.
    * 🌐 **Web:** Streamlit ile geliştirilmiş, hızlı analiz ve raporlama arayüzü.
* **3 Farklı Algoritma:** Greedy (Deterministik), Hill Climbing (Yerel Arama) ve Simulated Annealing (Global Arama).
* **Stokastik Simülasyon:** Algoritma yerleştirse bile, firmaların mülakatta %X ihtimalle reddetme durumu simüle edilebilir.
* **Görsel Analiz:** Matplotlib entegrasyonu ile başarı oranları ve skor karşılaştırmaları.

---

## 🚀 Canlı Demo (Web Arayüzü)

Projeyi bilgisayarınıza indirmeden, tarayıcı üzerinden test etmek için aşağıdaki butona tıklayın:

[👉 **Projeyi Canlı İncele (Streamlit Cloud)**](https://ahmet-yilmaz--intern-placement-simulation.streamlit.app)

---

## 🧠 Kullanılan Algoritmalar

### 1. Greedy (Açgözlü) Yaklaşım
* **Mantık:** Öğrencileri GNO'ya göre sıralar ve en başarılı öğrenciyi ilk tercihine yerleştirir.
* **Avantaj:** Çok hızlıdır (`O(N log N)`).
* **Dezavantaj:** Geriye dönük düzeltme yapmaz, yerel optimumda kalabilir.

### 2. Hill Climbing (Tepe Tırmanma)
* **Mantık:** Rastgele bir çözümle başlar. Rastgele iki öğrencinin yerini değiştirerek (Swap) daha yüksek bir memnuniyet puanı arar. Sadece "daha iyi" duruma gider.
* **Risk:** Yerel zirvelere (Local Maxima) takılıp kalabilir.

### 3. Simulated Annealing (Tavlama Benzetimi)
* **Mantık:** Hill Climbing'in gelişmiş halidir. Başlangıçta (Yüksek Sıcaklık) daha kötü çözümleri de kabul ederek yerel tuzaklardan kurtulur.
* **Formül:** Metropolis Kriteri (`P = e^(-ΔE/T)`) kullanılır. Global Optimum'a en yakın sonucu verir.

---

## 🛠️ Kurulum ve Çalıştırma

Projeyi yerel makinenizde çalıştırmak için aşağıdaki adımları izleyin.

### Gereksinimler
* Python 3.9 veya üzeri
* Git

### 1. Repoyu Klonlayın
```bash
git clone [https://github.com/KULLANICI_ADIN/Intern-Placement-Simulation.git](https://github.com/KULLANICI_ADIN/Intern-Placement-Simulation.git)
cd Intern-Placement-Simulation