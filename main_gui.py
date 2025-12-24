import sys
import time
import pandas as pd
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QTableWidget,
                             QTableWidgetItem, QHeaderView, QMessageBox, QTabWidget,
                             QFrame, QProgressBar, QStatusBar, QGraphicsDropShadowEffect,
                             QTextEdit, QLineEdit, QHeaderView)
from PyQt5.QtCore import Qt, pyqtSignal, QThread
from PyQt5.QtGui import QFont, QColor, QIntValidator

# --- MODÜL İMPORTLARI ---
# İsim çakışmalarını önlemek için fonksiyonları yeniden adlandırarak (alias) alıyoruz.
try:
    from veri_olustur import veri_seti_olustur
    from algo_greedy import greedy_atama, simulasyon_dongusu

    # Annealing algoritması
    from algo_heuristic_annealing import heuristic_atama as run_annealing, memnuniyet_skoru_hesapla

    # Hill Climbing algoritması
    from algo_heuristic_hill_climbing import heuristic_atama as run_hill_climbing

except ImportError as e:
    print(f"KRİTİK HATA: Modüller eksik! ({e})")


    # Programın çökmemesi için kukla (dummy) fonksiyonlar
    def veri_seti_olustur(*args, **kwargs):
        return pd.DataFrame(), pd.DataFrame()


    def greedy_atama(*args, **kwargs):
        return pd.DataFrame(), pd.DataFrame()


    def simulasyon_dongusu(*args, **kwargs):
        return pd.DataFrame(), pd.DataFrame(), []


    def run_annealing(*args, **kwargs):
        return pd.DataFrame(), pd.DataFrame()


    def run_hill_climbing(*args, **kwargs):
        return pd.DataFrame(), pd.DataFrame()


    def memnuniyet_skoru_hesapla(*args):
        return 0


# --- WORKER THREAD (GUI DONMAMASI İÇİN) ---
class HeuristicWorker(QThread):
    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal(object, object, float, str)
    error_signal = pyqtSignal(str)

    def __init__(self, target_function, ogrenciler, firmalar, iterasyon, algo_name):
        super().__init__()
        self.target_function = target_function
        self.ogrenciler = ogrenciler
        self.firmalar = firmalar
        self.iterasyon = iterasyon
        self.algo_name = algo_name

    def run(self):
        t_start = time.time()
        try:
            def callback(step):
                # Yüzdelik ilerleme hesabı
                if self.iterasyon > 0:
                    yuzde = int((step / self.iterasyon) * 100)
                    self.progress_signal.emit(yuzde)

            h_ogr, h_frm = self.target_function(
                self.ogrenciler,
                self.firmalar,
                iterasyon=self.iterasyon,
                step_callback=callback
            )
            sure = time.time() - t_start
            self.finished_signal.emit(h_ogr, h_frm, sure, self.algo_name)
        except Exception as e:
            self.error_signal.emit(str(e))


# --- GİRİŞ EKRANI ---
class GirisPenceresi(QWidget):
    pencere_gecis_sinyali = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sisteme Giriş")
        self.setFixedSize(450, 320)
        self.setStyleSheet("""
            QWidget { background-color: #2c3e50; }
            QLabel { color: white; }
            QPushButton { 
                background-color: #f1c40f; color: #2c3e50; border-radius: 6px; 
                font-size: 15px; font-weight: bold; padding: 12px; 
            }
            QPushButton:hover { background-color: #f39c12; }
        """)
        layout = QVBoxLayout()
        layout.addStretch()

        lbl_baslik = QLabel("STAJYER YERLEŞTİRME\nSİMÜLATÖRÜ")
        lbl_baslik.setAlignment(Qt.AlignCenter)
        lbl_baslik.setFont(QFont("Segoe UI", 20, QFont.Bold))

        lbl_alt = QLabel("2025-2026 Dönem Projesi\nPython Programlama")
        lbl_alt.setAlignment(Qt.AlignCenter)
        lbl_alt.setFont(QFont("Segoe UI", 11))
        lbl_alt.setStyleSheet("color: #bdc3c7;")

        self.btn_baslat = QPushButton("UYGULAMAYI BAŞLAT")
        self.btn_baslat.setCursor(Qt.PointingHandCursor)
        self.btn_baslat.setFixedWidth(220)
        self.btn_baslat.clicked.connect(self.tiklandi)

        layout.addWidget(lbl_baslik, alignment=Qt.AlignCenter)
        layout.addWidget(lbl_alt, alignment=Qt.AlignCenter)
        layout.addSpacing(25)
        layout.addWidget(self.btn_baslat, alignment=Qt.AlignCenter)
        layout.addStretch()
        self.setLayout(layout)

    def tiklandi(self):
        self.pencere_gecis_sinyali.emit()
        self.close()


# --- ANA ARAYÜZ ---
class ModernProjeArayuz(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("InternPlacement v6.0 - Proje Final")
        self.setGeometry(50, 50, 1500, 850)

        # Veri Değişkenleri
        self.df_ogrenciler = None
        self.df_firmalar = None

        self.sonuc_ogrenciler = None
        self.sonuc_firmalar = None

        # İstatistik Tutucular (Analiz İçin)
        self.scores = {"Greedy": 0, "HillClimb": 0, "Annealing": 0}
        self.times = {"Greedy": 0, "HillClimb": 0, "Annealing": 0}
        self.iters = {"Greedy": 1, "HillClimb": 3000, "Annealing": 5000}  # Varsayılan iterasyonlar

        self.init_ui()
        self.apply_styles()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # --- SOL MENÜ PANELİ ---
        left_panel = QFrame()
        left_panel.setObjectName("LeftPanel")
        left_panel.setFixedWidth(280)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(15, 40, 15, 20)
        left_layout.setSpacing(12)

        lbl_baslik = QLabel("STAJYER\nSİMÜLATÖRÜ")
        lbl_baslik.setObjectName("MenuTitle")
        lbl_baslik.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(lbl_baslik)

        lbl_yil = QLabel("2025 - 2026")
        lbl_yil.setObjectName("MenuSubtitle")
        lbl_yil.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(lbl_yil)

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("background-color: rgba(255,255,255,0.2);")
        left_layout.addWidget(line)

        # Girdi Alanları
        input_style = """
            QLineEdit { 
                background-color: rgba(255, 255, 255, 0.1); 
                color: white; border: 1px solid rgba(255, 255, 255, 0.3); 
                border-radius: 4px; padding: 5px; font-weight: bold;
            }
            QLabel { color: #bdc3c7; font-size: 12px; font-weight: bold; }
        """

        left_layout.addWidget(QLabel("Öğrenci Sayısı:", styleSheet=input_style))
        self.txt_ogrenci_sayisi = QLineEdit("150")
        self.txt_ogrenci_sayisi.setValidator(QIntValidator(1, 10000))
        self.txt_ogrenci_sayisi.setStyleSheet(input_style)
        left_layout.addWidget(self.txt_ogrenci_sayisi)

        left_layout.addWidget(QLabel("Firma Sayısı:", styleSheet=input_style))
        self.txt_firma_sayisi = QLineEdit("40")
        self.txt_firma_sayisi.setValidator(QIntValidator(1, 1000))
        self.txt_firma_sayisi.setStyleSheet(input_style)
        left_layout.addWidget(self.txt_firma_sayisi)
        left_layout.addSpacing(10)

        # Menü Butonları
        self.btn_veri = self.create_button("🎲  Veri Seti Oluştur", self.veri_uret_tikla)
        self.btn_greedy = self.create_button("🚀  Greedy Algoritması", self.greedy_calistir, False)
        self.btn_hill = self.create_button("⛰️  Hill Climbing", lambda: self.heuristic_baslat("HillClimb"), False)
        self.btn_annealing = self.create_button("🔥  Simulated Annealing", lambda: self.heuristic_baslat("Annealing"),
                                                False)
        self.btn_analiz = self.create_button("📊  Simülasyon & Analiz", self.analiz_sayfasini_ac, False)
        self.btn_reset = self.create_button("🔄  Sistemi Sıfırla", self.sistemi_sifirla)

        left_layout.addWidget(self.btn_veri)
        left_layout.addWidget(self.btn_greedy)
        left_layout.addWidget(self.btn_hill)
        left_layout.addWidget(self.btn_annealing)
        left_layout.addWidget(self.btn_analiz)
        left_layout.addStretch()
        left_layout.addWidget(self.btn_reset)
        main_layout.addWidget(left_panel)

        # --- SAĞ İÇERİK ALANI ---
        right_container = QWidget()
        right_container.setStyleSheet("background-color: #f4f6f9;")
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        # Üst Header
        header_frame = QFrame()
        header_frame.setStyleSheet("background-color: white; border-bottom: 1px solid #dcdcdc;")
        header_frame.setFixedHeight(60)
        header_frame.setGraphicsEffect(self.get_shadow())
        h_layout = QHBoxLayout(header_frame)
        h_layout.setContentsMargins(30, 0, 30, 0)

        self.lbl_page_title = QLabel("Kontrol Paneli")
        self.lbl_page_title.setFont(QFont("Segoe UI", 16, QFont.Bold))
        self.lbl_page_title.setStyleSheet("color: #2c3e50; border: none;")

        self.lbl_status = QLabel("Durum: Veri seti bekleniyor...")
        self.lbl_status.setStyleSheet("color: #7f8c8d; font-weight: bold;")

        h_layout.addWidget(self.lbl_page_title)
        h_layout.addStretch()
        h_layout.addWidget(self.lbl_status)
        right_layout.addWidget(header_frame)

        # Sayfalar (TabWidget)
        self.tabs_main = QTabWidget()
        self.tabs_main.tabBar().setVisible(False)  # Sekmeleri gizle, butonlarla geçeceğiz

        self.page_dashboard = QWidget()
        self.setup_dashboard_page(self.page_dashboard)
        self.tabs_main.addTab(self.page_dashboard, "Dashboard")

        self.page_analiz = QWidget()
        self.setup_analiz_page(self.page_analiz)
        self.tabs_main.addTab(self.page_analiz, "Analiz")

        right_layout.addWidget(self.tabs_main)

        # Progress Bar
        self.pbar = QProgressBar()
        self.pbar.setFixedHeight(5)
        self.pbar.setTextVisible(False)
        self.pbar.setStyleSheet(
            "QProgressBar {border:none; background:#e0e0e0;} QProgressBar::chunk {background:#27ae60;}")
        right_layout.addWidget(self.pbar)

        main_layout.addWidget(right_container)

    # --- DASHBOARD SAYFASI ---
    def setup_dashboard_page(self, parent):
        layout = QVBoxLayout(parent)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(20)

        # Kartlar
        dash_layout = QHBoxLayout()
        self.card_total = self.create_card("Toplam Öğrenci", "0", "#3498db")
        self.card_placed = self.create_card("Yerleşen", "0", "#27ae60")
        self.card_score = self.create_card("Memnuniyet Puanı", "0", "#f39c12")
        self.card_time = self.create_card("Son İşlem Süresi", "0 sn", "#9b59b6")
        dash_layout.addWidget(self.card_total)
        dash_layout.addWidget(self.card_placed)
        dash_layout.addWidget(self.card_score)
        dash_layout.addWidget(self.card_time)
        layout.addLayout(dash_layout)

        # Liste Tabloları
        self.tabs_data = QTabWidget()
        self.tabs_data.setStyleSheet(self.get_tab_style())

        self.tab_ogrenci = QTableWidget()
        self.tab_firma = QTableWidget()
        self.setup_table(self.tab_ogrenci)
        self.setup_table(self.tab_firma)

        self.tabs_data.addTab(self.tab_ogrenci, "📄 Öğrenci Listesi")
        self.tabs_data.addTab(self.tab_firma, "🏢 Firmalar ve Kontenjanlar")
        layout.addWidget(self.tabs_data)

    # --- ANALİZ SAYFASI (PROJE İSTERLERİNE GÖRE YENİ TASARIM) ---
    def setup_analiz_page(self, parent):
        layout = QHBoxLayout(parent)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)

        # --- SOL: SİMÜLASYON DÖNGÜSÜ (RED SENARYOSU) ---
        left_frame = QFrame()
        left_frame.setStyleSheet("background: white; border-radius: 8px;")
        left_frame.setGraphicsEffect(self.get_shadow())
        l_layout = QVBoxLayout(left_frame)

        lbl_sim_head = QLabel("🔄 Dinamik Red Simülasyonu")
        lbl_sim_head.setFont(QFont("Segoe UI", 12, QFont.Bold))
        lbl_sim_head.setStyleSheet("color: #2c3e50; border:none;")
        l_layout.addWidget(lbl_sim_head)

        lbl_sim_desc = QLabel(
            "Firmaların öğrencileri %10 olasılıkla reddettiği ve\nyeniden atamanın yapıldığı döngüsel süreç.")
        lbl_sim_desc.setStyleSheet("color: #7f8c8d; font-size: 11px; margin-bottom: 5px;")
        l_layout.addWidget(lbl_sim_desc)

        self.btn_sim_baslat = QPushButton("▶ Simülasyonu Başlat")
        self.btn_sim_baslat.setStyleSheet(
            "background-color: #e67e22; color: white; padding: 8px; font-weight: bold; border-radius: 5px;")
        self.btn_sim_baslat.setCursor(Qt.PointingHandCursor)
        self.btn_sim_baslat.clicked.connect(self.simulasyon_baslat)
        l_layout.addWidget(self.btn_sim_baslat)

        # Simülasyon Tablosu
        self.table_sim = QTableWidget()
        self.table_sim.setColumnCount(5)
        # Proje Raporu Madde 6: "Kaç öğrenci yerleşti, reddedildi, kalan kontenjan"
        self.table_sim.setHorizontalHeaderLabels(
            ["Tur", "Yerleşen", "Reddedilen", "Kalan Kont.", "Detay (Kim Reddedildi?)"])
        self.setup_table(self.table_sim)
        self.table_sim.horizontalHeader().setSectionResizeMode(4, QHeaderView.Stretch)  # Detay sütunu uzasın
        l_layout.addWidget(self.table_sim)

        layout.addWidget(left_frame, 55)  # %55 Genişlik

        # --- SAĞ: ALGORİTMA KARŞILAŞTIRMASI ---
        right_frame = QFrame()
        right_frame.setStyleSheet("background: white; border-radius: 8px;")
        right_frame.setGraphicsEffect(self.get_shadow())
        r_layout = QVBoxLayout(right_frame)

        lbl_comp_head = QLabel("🆚 Algoritma Karşılaştırma Raporu")
        lbl_comp_head.setFont(QFont("Segoe UI", 12, QFont.Bold))
        lbl_comp_head.setStyleSheet("color: #2c3e50; border:none;")
        r_layout.addWidget(lbl_comp_head)

        # Karşılaştırma Tablosu
        self.table_comp = QTableWidget()
        self.table_comp.setColumnCount(4)
        self.table_comp.setRowCount(4)  # 4 Kriter

        # Proje İsteri: Greedy vs Heuristik Kıyaslama Tablosu [cite: 55, 56, 57, 58]
        self.table_comp.setHorizontalHeaderLabels(["Kriter", "Greedy", "Hill Climbing", "Simulated Annealing"])
        self.table_comp.setVerticalHeaderLabels(["1", "2", "3", "4"])

        # Satır Başlıkları
        self.table_comp.setItem(0, 0, QTableWidgetItem("Memnuniyet Skoru"))
        self.table_comp.setItem(1, 0, QTableWidgetItem("Çözüm Süresi (sn)"))
        self.table_comp.setItem(2, 0, QTableWidgetItem("İşlem Sayısı (Iterasyon)"))
        self.table_comp.setItem(3, 0, QTableWidgetItem("Yerleşen Öğrenci"))

        self.setup_table(self.table_comp)
        r_layout.addWidget(self.table_comp)

        # Sonuç Yorum Alanı
        lbl_sonuc = QLabel("Sonuç Analizi:")
        lbl_sonuc.setStyleSheet("font-weight: bold; margin-top: 10px; color: #2c3e50;")
        r_layout.addWidget(lbl_sonuc)

        self.txt_sonuc = QTextEdit()
        self.txt_sonuc.setReadOnly(True)
        self.txt_sonuc.setPlaceholderText("Algoritmalar çalıştırıldığında karşılaştırma analizi burada görünecektir...")
        self.txt_sonuc.setStyleSheet(
            "background: #f8f9fa; border: 1px solid #ddd; padding: 10px; color: #333; font-size: 12px;")
        self.txt_sonuc.setMaximumHeight(150)
        r_layout.addWidget(self.txt_sonuc)

        layout.addWidget(right_frame, 45)  # %45 Genişlik

    # --- YARDIMCI FONKSİYONLAR ---
    def get_shadow(self):
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(15)
        shadow.setColor(QColor(0, 0, 0, 20))
        shadow.setOffset(0, 4)
        return shadow

    def get_tab_style(self):
        return """
            QTabWidget::pane { border: 1px solid #dcdcdc; background: white; border-radius: 5px; }
            QTabBar::tab { background: #ecf0f1; padding: 8px 20px; margin-right: 2px; border-top-left-radius: 5px; border-top-right-radius: 5px; }
            QTabBar::tab:selected { background: white; border-bottom: 3px solid #3498db; color: #3498db; font-weight: bold; }
        """

    def create_button(self, text, func, enabled=True):
        btn = QPushButton(text)
        btn.clicked.connect(func)
        btn.setEnabled(enabled)
        btn.setFixedHeight(50)
        btn.setCursor(Qt.PointingHandCursor)
        return btn

    def create_card(self, title, value, color):
        frame = QFrame()
        frame.setStyleSheet(f"background-color: white; border-left: 5px solid {color}; border-radius: 8px;")
        frame.setGraphicsEffect(self.get_shadow())
        l = QVBoxLayout(frame)
        lbl_t = QLabel(title)
        lbl_t.setStyleSheet("color: #7f8c8d; font-weight: bold; border:none;")
        lbl_v = QLabel(value)
        lbl_v.setObjectName("CardValue")
        lbl_v.setStyleSheet(f"color: {color}; font-size: 22px; font-weight: 800; border:none;")
        l.addWidget(lbl_t)
        l.addWidget(lbl_v)
        return frame

    def setup_table(self, table):
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        table.setAlternatingRowColors(True)
        table.setStyleSheet(
            "QTableWidget { background-color: white; border: none; gridline-color: #ecf0f1; } QHeaderView::section { background-color: #34495e; color: white; padding: 5px; }")

    def update_card(self, card, val):
        for lbl in card.findChildren(QLabel):
            if lbl.objectName() == "CardValue": lbl.setText(str(val))

    def tabloyu_doldur(self, df, table):
        if df is None: return
        table.setRowCount(df.shape[0])
        table.setColumnCount(df.shape[1])
        table.setHorizontalHeaderLabels(df.columns)
        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                table.setItem(i, j, QTableWidgetItem(str(df.iloc[i, j])))

    def apply_styles(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #f4f6f9; }
            #LeftPanel { background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #2c3e50, stop:1 #34495e); }
            #MenuTitle { color: #f1c40f; font-size: 18px; font-weight: 900; letter-spacing: 1px; margin-bottom: 5px; } 
            #MenuSubtitle { color: #bdc3c7; font-size: 13px; margin-bottom: 15px; }
            QPushButton { background-color: rgba(255, 255, 255, 0.05); color: white; border: none; padding-left: 15px; text-align: left; border-radius: 6px; font-size: 14px; }
            QPushButton:hover { background-color: rgba(255, 255, 255, 0.15); border-left: 4px solid #f1c40f; }
            QPushButton:disabled { color: #7f8c8d; }
        """)

    # --- EYLEMLER ---
    def veri_uret_tikla(self):
        try:
            o_sayi = int(self.txt_ogrenci_sayisi.text())
            f_sayi = int(self.txt_firma_sayisi.text())
            self.df_firmalar, self.df_ogrenciler = veri_seti_olustur(o_sayi, f_sayi)

            self.update_card(self.card_total, len(self.df_ogrenciler))
            self.tabloyu_doldur(self.df_ogrenciler, self.tab_ogrenci)
            self.tabloyu_doldur(self.df_firmalar, self.tab_firma)

            self.btn_greedy.setEnabled(True)
            self.btn_analiz.setEnabled(True)
            self.lbl_status.setText(f"Hazır: {o_sayi} Öğrenci, {f_sayi} Firma")
            self.tabs_main.setCurrentIndex(0)

            # Veri yeni üretildi, eski skorları sıfırla
            self.scores = {k: 0 for k in self.scores}
            self.times = {k: 0 for k in self.times}
            self.update_karsilastirma_tablosu()

        except Exception as e:
            QMessageBox.critical(self, "Hata", str(e))

    def greedy_calistir(self):
        self.lbl_status.setText("İşleniyor: Greedy Algoritması...")
        QApplication.processEvents()

        t1 = time.time()
        try:
            self.sonuc_ogrenciler, self.sonuc_firmalar = greedy_atama(self.df_ogrenciler, self.df_firmalar)
            t2 = time.time()

            sure = t2 - t1
            skor = memnuniyet_skoru_hesapla(self.sonuc_ogrenciler)

            # İstatistik Kaydı
            self.scores["Greedy"] = skor
            self.times["Greedy"] = sure

            yerlesen = self.sonuc_ogrenciler['Yerleştiği_Firma'].notna().sum()
            self.update_card(self.card_placed, yerlesen)
            self.update_card(self.card_score, skor)
            self.update_card(self.card_time, f"{sure:.4f} sn")

            self.tabloyu_doldur(self.sonuc_ogrenciler, self.tab_ogrenci)
            self.tabloyu_doldur(self.sonuc_firmalar, self.tab_firma)

            self.btn_hill.setEnabled(True)
            self.btn_annealing.setEnabled(True)
            self.lbl_status.setText("Tamamlandı: Greedy")

            # Karşılaştırma tablosunu anlık güncelle
            self.update_karsilastirma_tablosu()

        except Exception as e:
            QMessageBox.critical(self, "Hata", str(e))

    def heuristic_baslat(self, algo_tipi):
        self.lbl_status.setText(f"İşleniyor: {algo_tipi}...")
        self.btn_greedy.setEnabled(False)
        self.btn_hill.setEnabled(False)
        self.btn_annealing.setEnabled(False)
        self.pbar.setValue(0)

        # Hangi veri? (Greedy sonucu varsa ondan devam et, yoksa sıfırdan)
        input_ogr = self.sonuc_ogrenciler if self.sonuc_ogrenciler is not None else self.df_ogrenciler
        input_frm = self.sonuc_firmalar if self.sonuc_firmalar is not None else self.df_firmalar

        if algo_tipi == "HillClimb":
            target_func = run_hill_climbing
            iterasyon = self.iters["HillClimb"]
        else:
            target_func = run_annealing
            iterasyon = self.iters["Annealing"]

        self.worker = HeuristicWorker(target_func, input_ogr, input_frm, iterasyon, algo_tipi)
        self.worker.progress_signal.connect(lambda v: self.pbar.setValue(v))
        self.worker.finished_signal.connect(self.on_heuristic_finished)
        self.worker.error_signal.connect(self.on_heuristic_error)
        self.worker.start()

    def on_heuristic_finished(self, h_ogr, h_frm, sure, algo_name):
        self.sonuc_ogrenciler = h_ogr
        self.sonuc_firmalar = h_frm
        skor = memnuniyet_skoru_hesapla(h_ogr)

        self.scores[algo_name] = skor
        self.times[algo_name] = sure

        yerlesen = h_ogr['Yerleştiği_Firma'].notna().sum()
        self.update_card(self.card_placed, yerlesen)
        self.update_card(self.card_score, skor)
        self.update_card(self.card_time, f"{sure:.2f} sn")

        self.tabloyu_doldur(h_ogr, self.tab_ogrenci)
        self.tabloyu_doldur(h_frm, self.tab_firma)

        self.pbar.setValue(100)
        self.lbl_status.setText(f"Tamamlandı: {algo_name}")
        self.update_karsilastirma_tablosu()

        self.btn_greedy.setEnabled(True)
        self.btn_hill.setEnabled(True)
        self.btn_annealing.setEnabled(True)

    def on_heuristic_error(self, err):
        QMessageBox.critical(self, "Hata", err)
        self.btn_greedy.setEnabled(True)

    def simulasyon_baslat(self):
        if self.df_ogrenciler is None:
            QMessageBox.warning(self, "Uyarı", "Önce veri seti oluşturunuz.")
            return

        self.lbl_status.setText("Simülasyon: Red döngüsü çalışıyor...")
        self.table_sim.setRowCount(0)
        QApplication.processEvents()

        try:
            # Greedy'nin simülasyon fonksiyonunu çağır
            _, _, logs = simulasyon_dongusu(self.df_ogrenciler, self.df_firmalar)

            self.table_sim.setRowCount(len(logs))
            for i, log in enumerate(logs):
                # Tablo: Tur | Yerleşen | Reddedilen | Kalan | Detay
                self.table_sim.setItem(i, 0, QTableWidgetItem(str(log['Tur'])))
                self.table_sim.setItem(i, 1, QTableWidgetItem(str(log['Yerleşen'])))
                self.table_sim.setItem(i, 2, QTableWidgetItem(str(log['Reddedilen'])))
                self.table_sim.setItem(i, 3, QTableWidgetItem(str(log['Kalan_Kontenjan'])))

                # Detay sütunu (mouse üzerine gelince tamamını gösterir)
                item_detay = QTableWidgetItem(str(log['Red_Detay']))
                item_detay.setToolTip(str(log['Red_Detay']))
                self.table_sim.setItem(i, 4, item_detay)

            self.lbl_status.setText("Simülasyon Tamamlandı.")

        except Exception as e:
            QMessageBox.critical(self, "Simülasyon Hatası", str(e))

    def update_karsilastirma_tablosu(self):
        # 1. Greedy Verileri
        self.table_comp.setItem(0, 1, QTableWidgetItem(str(self.scores["Greedy"])))
        self.table_comp.setItem(1, 1, QTableWidgetItem(f"{self.times['Greedy']:.4f}"))
        self.table_comp.setItem(2, 1, QTableWidgetItem("1 (Tek Geçiş)"))
        if self.sonuc_ogrenciler is not None:
            g_yer = self.sonuc_ogrenciler['Yerleştiği_Firma'].notna().sum() if self.scores["Greedy"] > 0 else "-"
            self.table_comp.setItem(3, 1, QTableWidgetItem(str(g_yer)))

        # 2. Hill Climbing Verileri
        hc_score = self.scores["HillClimb"]
        self.table_comp.setItem(0, 2, QTableWidgetItem(str(hc_score)))
        self.table_comp.setItem(1, 2, QTableWidgetItem(f"{self.times['HillClimb']:.2f}"))
        self.table_comp.setItem(2, 2, QTableWidgetItem(str(self.iters['HillClimb']) if hc_score > 0 else "-"))
        # (Heuristik sonucunda yerleşen sayısı genelde aynı kalır ama değişebilir, burada göstermelik tire)
        self.table_comp.setItem(3, 2, QTableWidgetItem("-" if hc_score == 0 else "Sabit/Artan"))

        # 3. Annealing Verileri
        sa_score = self.scores["Annealing"]
        self.table_comp.setItem(0, 3, QTableWidgetItem(str(sa_score)))
        self.table_comp.setItem(1, 3, QTableWidgetItem(f"{self.times['Annealing']:.2f}"))
        self.table_comp.setItem(2, 3, QTableWidgetItem(str(self.iters['Annealing']) if sa_score > 0 else "-"))
        self.table_comp.setItem(3, 3, QTableWidgetItem("-" if sa_score == 0 else "Sabit/Artan"))

        # Sonuç Analizi Yazısı
        greedy = self.scores["Greedy"]
        best_heuristic = max(self.scores["HillClimb"], self.scores["Annealing"])

        if best_heuristic > 0 and greedy > 0:
            fark = best_heuristic - greedy
            yuzde = (fark / greedy) * 100

            if fark > 0:
                msg = (f"✅ BAŞARILI: Heuristik algoritmalar Greedy sonucunu iyileştirdi.\n"
                       f"Memnuniyet Puan Artışı: +{fark} puan (%{yuzde:.2f})\n"
                       f"Bu sonuç, yerel aramaların (local search) Greedy'nin takıldığı lokal optimumdan kurtulduğunu gösterir.")
                self.txt_sonuc.setStyleSheet("color: green; font-weight: bold; background: #e8f5e9;")
            elif fark == 0:
                msg = ("ℹ️ EŞİT SONUÇ: Greedy algoritması zaten çok iyi bir sonuç bulmuş olabilir veya\n"
                       "Heuristik iterasyon sayısı daha iyi bir sonuç bulmak için yetersiz kaldı.")
                self.txt_sonuc.setStyleSheet("color: #d35400; font-weight: bold; background: #fcf3cf;")
            else:
                msg = "⚠️ HATA: Heuristik skor daha düşük. (Normalde olmaması gerekir, kod mantığını kontrol edin.)"

            self.txt_sonuc.setText(msg)

    def analiz_sayfasini_ac(self):
        self.lbl_page_title.setText("Analiz ve Raporlama")
        self.tabs_main.setCurrentIndex(1)
        self.update_karsilastirma_tablosu()

    def sistemi_sifirla(self):
        self.df_ogrenciler = None
        self.sonuc_ogrenciler = None
        self.scores = {k: 0 for k in self.scores}
        self.times = {k: 0 for k in self.times}

        self.tab_ogrenci.setRowCount(0)
        self.tab_firma.setRowCount(0)
        self.table_sim.setRowCount(0)

        self.update_card(self.card_total, "0")
        self.update_card(self.card_placed, "0")
        self.update_card(self.card_score, "0")
        self.update_card(self.card_time, "0")

        self.btn_greedy.setEnabled(False)
        self.btn_hill.setEnabled(False)
        self.btn_annealing.setEnabled(False)
        self.btn_analiz.setEnabled(False)

        self.pbar.setValue(0)
        self.tabs_main.setCurrentIndex(0)
        self.lbl_page_title.setText("Kontrol Paneli")
        self.txt_sonuc.clear()
        self.lbl_status.setText("Sistem Sıfırlandı")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    giris = GirisPenceresi()
    arayuz = ModernProjeArayuz()

    giris.pencere_gecis_sinyali.connect(arayuz.show)
    giris.show()
    sys.exit(app.exec_())