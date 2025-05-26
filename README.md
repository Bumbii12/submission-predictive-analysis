
# Laporan Proyek Machine Learning - Ulfa Stevi Juliana

## Domain Proyek

**Analisis sentimen** telah menjadi salah satu topik utama dalam Natural Language Processing (NLP) dan data science secara umum, terutama dalam memahami opini pelanggan terhadap produk atau layanan. Dalam konteks ini, ulasan pelanggan di situs perjalanan seperti **Trip Advisor** menjadi sumber data yang sangat berharga untuk mengidentifikasi kepuasan pelanggan terhadap layanan hotel.

Masalah ini penting untuk diselesaikan karena:
- Memberikan wawasan bisnis yang berguna bagi manajemen hotel untuk meningkatkan layanan mereka.
- Membantu calon pelanggan membuat keputusan berdasarkan ulasan.
- Mendorong pemanfaatan data tidak terstruktur (teks) dalam sistem pendukung keputusan.

### Referensi:
- M. Liu, “Sentiment Analysis and Opinion Mining,” Synthesis Lectures on Human Language Technologies, vol. 5, no. 1, pp. 1–167, 2012. [Online]. Available: https://www.morganclaypool.com/doi/abs/10.2200/S00416ED1V01Y201204HLT016

## Business Understanding

### Problem Statements
1. Bagaimana mengklasifikasikan ulasan pengguna ke dalam kategori sentimen positif dan negatif secara otomatis?
2. Algoritma mana yang paling efektif untuk analisis sentimen terhadap ulasan Trip Advisor?

### Goals
1. Membangun model machine learning untuk mengklasifikasikan ulasan hotel berdasarkan sentimen.
2. Mengevaluasi dan membandingkan performa beberapa algoritma untuk menemukan model terbaik.

### Solution Statements
- Menggunakan dua algoritma: **Logistic Regression** dan **Multinomial Naive Bayes** sebagai baseline.
- Melakukan tuning parameter dan menggunakan teknik **TF-IDF vectorization** untuk meningkatkan performa model.
- Mengukur performa dengan metrik **accuracy, precision, recall, dan F1-score**.

## Data Understanding

Dataset yang digunakan adalah **Trip Advisor Hotel Reviews** yang memuat ulasan pengguna dan label sentimen (0 = negatif, 1 = positif). Sumber data berasal dari Kaggle: [Trip Advisor Hotel Reviews](https://www.kaggle.com/datasets/andrewmvd/trip-advisor-hotel-reviews).

### Variabel dalam dataset:
- **Review**: Teks ulasan hotel dari pengguna.
- **Rating**: Skor atau label sentimen (0 atau 1).

EDA yang dilakukan termasuk:
- Menampilkan distribusi label.
- Visualisasi kata paling sering muncul menggunakan WordCloud.
- Analisis panjang teks.

## Data Preparation

Tahapan yang dilakukan:
- **Lowercasing**: Mengubah semua huruf menjadi kecil.
- **Cleaning**: Menghapus karakter non-alfabet, angka, dan tanda baca dengan regex.
- **Tokenisasi dan lemmatization** menggunakan spaCy.
- **Vectorization**: Mengubah teks menjadi representasi numerik dengan TF-IDF dan CountVectorizer.

Alasan data preparation:
- Mengurangi noise pada data teks.
- Meningkatkan kualitas fitur sebelum dimasukkan ke model machine learning.

## Modeling

Model yang digunakan:
1. **Multinomial Naive Bayes**
   - Cocok untuk data teks yang bersifat multinomial seperti word count/TF-IDF.
2. **Logistic Regression**
   - Sering digunakan untuk klasifikasi biner.

Kedua model dilatih pada data TF-IDF hasil preprocessing. Tidak dilakukan hyperparameter tuning secara eksplisit.

### Kelebihan dan kekurangan:
- **Naive Bayes**: Cepat dan efektif untuk teks, tapi mengasumsikan independensi fitur.
- **Logistic Regression**: Lebih fleksibel, namun bisa lebih lambat dan sensitif terhadap fitur korup.

## Evaluation

Metrik yang digunakan:
- **Accuracy**: Proporsi prediksi benar terhadap total data.
- **Precision**: Proporsi prediksi positif yang benar.
- **Recall**: Proporsi kasus positif yang berhasil teridentifikasi.
- **F1-score**: Harmonik rata-rata precision dan recall.

### Hasil evaluasi:
- **Naive Bayes Accuracy**: ~89%
- **Logistic Regression Accuracy**: ~90%
- Model terbaik: **Logistic Regression**, karena memiliki akurasi dan F1-score sedikit lebih tinggi.

---
