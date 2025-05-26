# Laporan Proyek Analisis Sentimen dengan BERT – TripAdvisor Hotel Reviews

## 1. Domain Proyek

### Latar Belakang

Industri perhotelan sangat bergantung pada kepuasan pelanggan. Seiring perkembangan teknologi dan kemudahan akses informasi, calon tamu hotel kini semakin mengandalkan ulasan dari pengguna sebelumnya untuk menentukan pilihan mereka. Ulasan-ulasan ini tidak hanya memuat opini, kritik, dan pengalaman pribadi, namun juga mencerminkan sentimen yang bisa dikategorikan sebagai positif atau negatif.

Mengingat banyaknya jumlah ulasan yang tersedia secara online, proses analisis manual sangat tidak efisien. Oleh karena itu, dibutuhkan pendekatan otomatis berbasis Machine Learning untuk memahami dan mengklasifikasikan sentimen pelanggan. Dalam proyek ini, saya mengembangkan model analisis sentimen menggunakan pendekatan Deep Learning berbasis transformer, yaitu BERT (Bidirectional Encoder Representations from Transformers). Dataset yang digunakan berasal dari TripAdvisor, berisi 20.491 ulasan dan rating hotel.

## 2. Business Understanding

### Permasalahan Bisnis

- Bagaimana cara mengevaluasi sentimen pelanggan secara cepat dari ribuan ulasan?
- Bagaimana mengubah data teks ulasan menjadi wawasan strategis bagi pihak manajemen hotel?

### Tujuan

- Membangun sistem otomatis yang dapat mengklasifikasikan ulasan pelanggan menjadi sentimen positif atau negatif.
- Memberikan insight strategis untuk perbaikan layanan dan pengambilan keputusan yang berbasis data.

### Manfaat

- **Efisiensi analisis ulasan** dalam jumlah besar.
- **Identifikasi area perbaikan layanan** berdasarkan opini pelanggan.
- **Penguatan strategi pemasaran** dengan memahami persepsi pelanggan.
- **Keunggulan kompetitif**, melalui pemahaman mendalam terhadap kepuasan tamu hotel.

### Solusi yang Diterapkan

1. **Preprocessing Data**  
   - Menghapus karakter tidak relevan seperti HTML, URL, angka, tanda baca, emoji, dan spasi berlebih.  
   - Menurunkan huruf kapital dan menghapus stopwords menggunakan NLTK.  
   - Mengonversi rating menjadi dua kelas sentimen:  
     - **Positif**: rating 4, 5  
     - **Negatif**: rating 1, 2, 3

2. **Modeling dengan BERT**  
   - Menggunakan tokenizer dan model pre-trained BERT untuk klasifikasi biner.  
   - Tokenisasi dengan padding dan truncation untuk panjang maksimum 128 token.  
   - Pembagian data ke dalam set pelatihan, validasi, dan pengujian.

3. **Training**  
   - Model dilatih menggunakan PyTorch dan optimizer AdamW.  
   - Hyperparameter: learning rate 5e-5, batch size 16, dan epoch sebanyak 3.

4. **Evaluasi**  
   - Evaluasi dilakukan dengan akurasi, precision, recall, dan F1-score.  
   - Model diuji pada dataset terpisah untuk mengukur generalisasi.

## 3. Data Understanding

### Informasi Dataset

- Sumber: [Kaggle - TripAdvisor Hotel Reviews](https://www.kaggle.com/datasets/andrewmvd/trip-advisor-hotel-reviews)  
- Format: CSV  
- Jumlah data: 20.491 baris dan 2 kolom (Review, Rating)

Kolom:
- **Review**: berisi teks ulasan
- **Rating**: angka 1 sampai 5 yang menunjukkan penilaian pelanggan

## 4. Data Preparation

Langkah-langkah persiapan data:

1. **Pembersihan Teks**  
   - Menghapus HTML, URL, angka, tanda baca, emoji  
   - Lowercasing dan stopword removal

2. **Label Sentimen**  
   - Positif (1): rating 4, 5  
   - Negatif (0): rating 1, 2, 3

3. **Pembagian Data**  
   - Stratified sampling: 70% data pelatihan dan 30% data testing

4. **Tokenisasi**  
   - Menggunakan tokenizer BERT  
   - Padding dan truncation ke 128 token

5. **Konversi ke Tensor Dataset**  
   - Dataset diubah menjadi objek tensor PyTorch agar dapat digunakan dalam proses pelatihan model.

## 5. Modeling

### Arsitektur Model

- Menggunakan model `bert-base-uncased` dari HuggingFace
- Klasifikasi biner menggunakan linear layer pada akhir BERT
- Tokenisasi dilakukan dengan `BertTokenizer`
- Parameter awal model BERT dipertahankan (fine-tuning dilakukan secara parsial)

### Hyperparameter dan Konfigurasi

| Parameter        | Nilai     |
|------------------|-----------|
| Learning Rate    | 5e-5      |
| Batch Size       | 16        |
| Epoch            | 3         |
| Optimizer        | AdamW     |
| Scheduler        | Linear    |
| Loss Function    | CrossEntropyLoss  |

### Proses Training

- Input: input_ids dan attention_mask
- Loss: Binary Cross Entropy
- Optimizer: AdamW
- Performa divalidasi setiap epoch
- Checkpoint model disimpan jika validasi meningkat

## 6. Evaluasi

### Hasil Akurasi

- Akurasi pelatihan mencapai ~97%
- Akurasi validasi mencapai ~94%

![Model Accuracy Plot]()

### Confusion Matrix

![Confusion Matrix]()

- True Positive: 4264  
- True Negative: 1248  
- False Positive: 264  
- False Negative: 372

### Classification Report

| Sentimen | Precision | Recall | F1-Score | Support |
|----------|-----------|--------|----------|---------|
| Negatif  | 0.83      | 0.77   | 0.80     | 1620     |
| Positif  | 0.92      | 0.94   | 0.93     | 4528    |
| **Akurasi** |     |      | **0.90**  | 6148    |

### Analisis Hasil

- Model menunjukkan performa sangat baik dalam mengklasifikasikan ulasan positif (F1 93%)
- Performa pada kelas negatif juga cukup solid dengan F1-score 80%
- Skor weighted average F1 mencapai 90%, menunjukkan model sangat andal pada data uji

## 7. Kesimpulan

Model BERT yang dikembangkan berhasil melakukan klasifikasi sentimen ulasan hotel dengan akurasi tinggi. Evaluasi menunjukkan bahwa model sangat andal dalam mendeteksi sentimen positif dan cukup baik dalam menangkap sentimen negatif. Sistem ini dapat diandalkan untuk membantu hotel memahami umpan balik pelanggan dan meningkatkan kualitas layanan berdasarkan data.
