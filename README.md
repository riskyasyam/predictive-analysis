# Laporan Proyek Machine Learning - Muhammad Rizky Asyam Haidar

## Domain Proyek : Kesehatan (Penyakit Jantung)

![Grafik Akurasi Model](heart-disease.jpg)

Penyakit jantung merupakan penyebab kematian tertinggi di Indonesia. Salah satu bentuk serius dari penyakit ini adalah serangan jantung (heart attack), yang sering kali tidak terdeteksi sejak dini. Dalam konteks layanan kesehatan masyarakat, kemampuan untuk memprediksi risiko serangan jantung berdasarkan data medis dan gaya hidup dapat memberikan intervensi lebih awal dan mengurangi angka kematian.
![Referensi 1](ss1.png)

Menurut laporan WHO dan Kementerian Kesehatan RI, faktor risiko seperti hipertensi, kolesterol tinggi, kebiasaan merokok, obesitas, serta gaya hidup tidak sehat sangat berkontribusi terhadap penyakit kardiovaskular.
![Referensi 2](ss2.png)

Model machine learning dalam proyek ini bertujuan untuk membantu menyelesaikan masalah tingginya angka kematian akibat serangan jantung dengan cara menganalisis data medis dan gaya hidup pasien. Dengan mempelajari pola dari data tersebut, model dapat memprediksi kemungkinan seorang individu mengalami serangan jantung. Kemampuan prediksi ini sangat berharga karena memungkinkan identifikasi dini individu berisiko tinggi, sehingga intervensi medis dan perubahan gaya hidup dapat dilakukan lebih awal untuk mencegah terjadinya serangan jantung atau mengurangi dampaknya. Ini berpotensi meningkatkan kualitas layanan kesehatan masyarakat dan menurunkan angka kematian akibat penyakit ini.

## Referensi:
- Kementerian Kesehatan RI, "Situasi Penyakit Jantung di Indonesia", 2018.
- WHO, Cardiovascular Diseases (CVDs), 2021.

## Business Understanding

### Problem Statements

- **PS1** Bagaimana cara memprediksi kemungkinan serangan jantung berdasarkan data kesehatan individu?
- **PS2** Algoritma machine learning mana yang paling efektif untuk mengklasifikasikan risiko serangan jantung dalam dataset ini?

### Goals

- **G1** Membangun model prediksi risiko serangan jantung dengan akurasi dan performa metrik yang layak.
- **G2** Membandingkan performa beberapa algoritma machine learning dalam mengklasifikasikan risiko tersebut.

    ### Solution statements
    - Menerapkan empat algoritma klasifikasi: Logistic Regression, Support Vector Machine (SVM), K-Nearest Neighbors (KNN), dan XGBoost.
    - Melakukan balancing pada kelas target menggunakan `class_weight='balanced'` untuk Logistic Regression.
    - Mengevaluasi performa dengan metrik: accuracy, precision, recall, dan F1-score.

## Data Understanding
Pada tahap ini, dilakukan pemahaman mendalam terhadap dataset yang akan digunakan untuk membangun model.

- Tautan Sumber Data: https://www.kaggle.com/datasets/ankushpanday2/heart-attack-prediction-in-indonesia
- Nama Dataset: Heart Attack Prediction in Indonesia
- Informasi Dataset Awal:
   - Jumlah Data: Dataset ini terdiri dari 158.355 sampel (baris).
   - Jumlah Fitur (Kolom): Terdapat 28 kolom, termasuk fitur target.
- Kondisi Data Awal:
   - Missing Values: Berdasarkan pemeriksaan awal pada data yang dimuat, tidak ditemukan nilai yang hilang (missing values) pada dataset.
   - Duplikasi Data: Tidak terdapat duplicate data

### Uraian Seluruh Fitur pada Data :
Dataset ini mencakup berbagai informasi terkait kondisi medis, demografi, dan gaya hidup responden, yang semuanya berpotensi menjadi prediktor serangan jantung.
- **age**: Usia responden (dalam tahun).
- **gender**: Jenis kelamin responden (`Male`/`Female`).
- **region**: Wilayah tempat tinggal responden.
- **income_level**: Tingkat pendapatan responden (`Low`, `Medium`, `High`).
- **hypertension**: Riwayat hipertensi (1 = ya, 0 = tidak).
- **diabetes**: Riwayat diabetes (1 = ya, 0 = tidak).
- **cholesterol_level**: Tingkat kolesterol total (numerik).
- **obesity**: Status obesitas (1 = obesitas, 0 = tidak).
- **waist_circumference**: Lingkar pinggang (dalam cm).
- **family_history**: Riwayat keluarga dengan penyakit jantung (1 = ada, 0 = tidak).
- **smoking_status**: Status merokok (`Never`, `Former`, `Current`).
- **alcohol_consumption**: Konsumsi alkohol (`Yes`/`No`).
- **physical_activity**: Aktivitas fisik reguler (`Yes`/`No`).
- **dietary_habits**: Pola makan sehat (`Healthy`, `Unhealthy`).
- **air_pollution_exposure**: Tingkat paparan polusi udara (`Low`, `Moderate`, `High`).
- **stress_level**: Tingkat stres responden (`Low`, `Moderate`, `High`).
- **sleep_hours**: Rata-rata jam tidur per malam (dalam jam).
- **blood_pressure_systolic**: Tekanan darah sistolik (mmHg).
- **blood_pressure_diastolic**: Tekanan darah diastolik (mmHg).
- **fasting_blood_sugar**: Kadar gula darah puasa (mg/dL).
- **cholesterol_hdl**: Kadar kolesterol HDL (mg/dL).
- **cholesterol_ldl**: Kadar kolesterol LDL (mg/dL).
- **triglycerides**: Kadar trigliserida (mg/dL).
- **EKG_results**: Hasil pemeriksaan EKG (`Normal`, `Abnormal`, `Borderline`).
- **previous_heart_disease**: Riwayat penyakit jantung sebelumnya (1 = ya, 0 = tidak).
- **medication_usage**: Konsumsi obat terkait jantung (1 = ya, 0 = tidak).
- **participated_in_free_screening**: Partisipasi dalam skrining gratis (1 = ya, 0 = tidak).
- **heart_attack**: Target variabel: apakah pernah mengalami serangan jantung (1 = ya, 0 = tidak).

## Exploratory Data Analysis dan Visualisasi Data
- **Missing Values** = Tidak ada missing values (0)
- **Pembagian Data** : Data dibagi menjadi 2 jenis yaitu numeric_features dan categorical_features
- **Deteksi Outliers Numeric Features dengan IQR** : 
   ![Deteksi Outliers](outliers.png)
   - Grafik ini membandingkan distribusi fitur numerik sebelum (warna merah muda) dan sesudah (warna hijau) penghapusan outlier. Penghapusan outlier bertujuan untuk memperbaiki distribusi data dan menghindari pengaruh negatif nilai ekstrem terhadap model.
   - Setelah dilakukan deteksi outliers terdapat beberapa outliers seperti di age >85 sangat ekstrim sehingga terdeteksi outliers. lanjut EDA Univariate

### Exploratory Data Analysis (EDA) Univariate
- **Distribusi Variabel Kategorikal** : 
   ![Distribusi Kategorikal](distribusi-kategorikal.png)
   - **gender** : Distribusi gender relatif seimbang antara pria dan wanita.
   - **smoking_status** : Mayoritas responden adalah yang tidak pernah merokok dan Diikuti oleh perokok saat ini dan mantan perokok.
   - **physical_activity** : Sebagian besar individu memiliki aktivitas fisik sedang dan rendah dan Aktivitas fisik tinggi hanya mencakup sebagian kecil responden.
   - **dietary_habits** : Sebagian besar responden memiliki kebiasaan makan yang tidak sehat.
   - **air_pollution_exposure** : Paparan polusi udara sedang paling dominan, diikuti oleh rendah dan tinggi.
   - **stress_level** : Mayoritas memiliki tingkat stres sedang.dan Tingkat stres rendah dan tinggi lebih sedikit.
   - **hypertension, diabetes, obesity, previous_heart_disease, medication_usage** : Sebagian besar responden tidak memiliki kondisi tersebut (label `0`). dan Rasio yang memiliki kondisi (`1`) relatif kecil, menunjukkan imbalance data.
- **Distribusi Variabel Numerical** : 
   ![Distribusi Numerical](distribusi-numerical.png)
   - **age** : Distribusi usia menyerupai distribusi normal dengan puncak sekitar usia 60 tahun.
   - **cholesterol_level** : Distribusi simetris dengan puncak di sekitar 200 mg/dL.
   - **waist_circumference** : Bentuk distribusi normal dengan rata-rata sekitar 90–100 cm.
   - **sleep_hours** : Distribusi tidak normal. dan Mayoritas tidur antara 6–8 jam, ada lonjakan pada 9 jam.
   - **blood_pressure_systolic` dan `blood_pressure_diastolic**: Kedua variabel memiliki distribusi mendekati normal. dan Tekanan sistolik rata-rata di sekitar 130 mmHg dan diastolik sekitar 80 mmHg.
   - **fasting_blood_sugar** : Distribusi skewed ke kanan (positif). Konsentrasi utama antara 90–120 mg/dL.
   - **cholesterol_hdl`, `cholesterol_ldl** : Distribusi cenderung normal. HDL sekitar 50 mg/dL, LDL sekitar 130 mg/dL.
   - **triglycerides**
    Distribusi agak skewed ke kanan. dan Nilai trigliserida kebanyakan di bawah 200 mg/dL.

### Exploratory Data Analysis (EDA) Multivariate
- **Distribusi Serangan Jantung Berdasarkan Fitur Kategorikal**
   ![Multivariate Kategori](multivarite-kategori.png)
   - **Gender**  : Laki-laki memiliki jumlah kasus serangan jantung yang sedikit lebih tinggi dibanding perempuan.   Ini mengindikasikan gender berpotensi menjadi faktor risiko.
   - **Smoking Status**  : Individu yang merokok (current) memiliki proporsi serangan jantung yang lebih tinggi.   Namun, jumlah terbesar berasal dari kelompok tidak merokok (never), mengindikasikan ada faktor lain yang juga berpengaruh.
   - **Physical Activity**  : Tingkat aktivitas tinggi cenderung terkait dengan lebih sedikit kasus serangan jantung.   Sebaliknya, aktivitas rendah memiliki proporsi serangan jantung lebih tinggi.
   - **Dietary Habits**  : Pola makan tidak sehat (Unhealthy) berkorelasi dengan lebih banyak kasus serangan jantung.   Diet sehat (Healthy) tampaknya menjadi faktor protektif.
   - **Air Pollution Exposure**  : Paparan polusi tinggi berkaitan dengan lebih banyak kasus serangan jantung dibanding paparan rendah. 
   - **Stress Level**  : Stres tinggi berhubungan erat dengan frekuensi serangan jantung yang lebih tinggi.  Hal ini menunjukkan pentingnya faktor psikologis.
   - **Hypertension**  : Penderita hipertensi secara signifikan memiliki lebih banyak kasus serangan jantung.
   - **Diabetes**  : Penderita diabetes juga menunjukkan kecenderungan lebih tinggi mengalami serangan jantung.
   - **Obesity** :  Kasus serangan jantung lebih banyak terjadi pada individu dengan obesitas.
   - **Previous Heart Disease** : Riwayat penyakit jantung sebelumnya adalah indikator kuat terhadap serangan jantung berulang.
   - **Medication Usage** : Penggunaan obat memiliki jumlah kasus serangan jantung yang relatif tinggi, menandakan pasien berisiko tinggi sedang diobati.
- **Korelasi Fitur Numeric** :
   ![Korelasi Matrix](korelasi.png)
   - Tidak ada korelasi tinggi antar fitur numerik, yang terlihat dari nilai korelasi mendekati 0.
   - Korelasi tertinggi terjadi antara:
      - blood_pressure_systolic dan blood_pressure_diastolic: ~0.5
      - cholesterol_level dan cholesterol_ldl: ~0.4
   - Sebagian besar fitur berdistribusi independen, artinya masing-masing bisa menyumbang informasi unik ke dalam model prediksi.

## Data Preparation
**Mengapa Diperlukan Data Preparation** :
- Kualitas Data Mempengaruhi Hasil Model
- Beberapa algoritma membutuhkan data dalam format numerik dan berskala.
- Menyeimbangkan Data (Handling Imbalance)
- Meningkatkan Performa dan Akurasi Model

**Penghapusan Fitur (Feature Dropping)**
Beberapa fitur dihilangkan karena dianggap kurang relevan untuk prediksi serangan jantung dalam konteks model ini, atau karena alasan privasi dan potensi bias. Fitur yang dihapus adalah: 
- region, 
- income_level,
- family_history,
- alcohol_consumption,
- EKG_results, 
- participated_in_free_screening.
Jumlah fitur setelah drop: 22 (termasuk target).

**Pengananan Outliers**
Outlier pada fitur numerik dideteksi menggunakan metode Interquartile Range (IQR). Nilai yang berada di luar rentang (Q1−1.5 timesIQR) hingga (Q3+1.5 timesIQR) diidentifikasi sebagai outlier dan kemudian dihapus dari dataset.

- Tujuan: Menghindari pengaruh negatif nilai ekstrem terhadap performa model dan memperbaiki distribusi data.
- Visualisasi Perbandingan Distribusi: 
   ![Perbandingan Distribusi Outliers](after.png)
- Setelah dilakukan penghapusan outliers data sebagai berikut : 
   - **`age`**: Distribusi usia menunjukkan beberapa nilai ekstrem di atas 85 tahun yang dikategorikan sebagai outlier. Setelah dihapus, distribusi menjadi lebih simetris.
  
   - **`cholesterol_level`**, **`cholesterol_hdl`**, **`cholesterol_ldl`**: Masing-masing menunjukkan distribusi mendekati normal, namun memiliki beberapa nilai sangat tinggi. Penghapusan outlier memperhalus kurva distribusi.
  
   - **`waist_circumference`**: Beberapa nilai sangat rendah atau sangat tinggi dikategorikan sebagai outlier. Setelah dihapus, data lebih terkonsentrasi di tengah distribusi.

   - **`sleep_hours`**: Terdapat lonjakan tidak wajar pada jam tidur ekstrem (misalnya > 9 jam). Setelah penghapusan, distribusi menjadi lebih wajar.

   - **`blood_pressure_systolic` & `blood_pressure_diastolic`**: Keduanya menunjukkan distribusi normal dengan sedikit outlier ekstrem di bawah/atas ambang klinis.

   - **`fasting_blood_sugar`**: Menampilkan distribusi skewed kanan, dengan banyak outlier di atas 160 mg/dL. Setelah penghapusan, distribusi menjadi lebih ramping.

   - **`triglycerides`**: Outlier yang sangat tinggi mempengaruhi bentuk distribusi. Setelah dibersihkan, histogram menjadi lebih simetris.


**One Hot Encoding**
| gender_Male | smoking_status_Never | smoking_status_Past | physical_activity_Low | physical_activity_Moderate | dietary_habits_Unhealthy | air_pollution_exposure_Low | air_pollution_exposure_Moderate | stress_level_Low | stress_level_Moderate | hypertension_1 | diabetes_1 | obesity_1 | previous_heart_disease_1 | medication_usage_1 |
|-------------|----------------------|----------------------|------------------------|-----------------------------|--------------------------|----------------------------|-------------------------------|-------------------|------------------------|----------------|-------------|------------|--------------------------|---------------------|
| 1.0         | 1.0                  | 0.0                  | 0.0                    | 0.0                         | 1.0                      | 0.0                        | 1.0                           | 0.0               | 1.0                    | 0.0            | 1.0         | 0.0        | 0.0                      | 0.0                 |
| 0.0         | 0.0                  | 1.0                  | 0.0                    | 1.0                         | 0.0                      | 0.0                        | 0.0                           | 0.0               | 0.0                    | 0.0            | 0.0         | 0.0        | 1.0                      | 0.0                 |
| 0.0         | 0.0                  | 1.0                  | 0.0                    | 1.0                         | 0.0                      | 1.0                        | 0.0                           | 1.0               | 0.0                    | 0.0            | 0.0         | 1.0        | 0.0                      | 1.0                 |
| 1.0         | 1.0                  | 0.0                  | 0.0                    | 1.0                         | 1.0                      | 1.0                        | 0.0                           | 0.0               | 0.0                    | 1.0            | 0.0         | 0.0        | 0.0                      | 1.0                 |
| 1.0         | 0.0                  | 0.0                  | 0.0                    | 1.0                         | 1.0                      | 0.0                        | 0.0                           | 0.0               | 1.0                    | 1.0            | 0.0         | 0.0        | 1.0                      | 0.0                 |

Fitur-fitur kategorikal yang tersisa dalam df_no_outliers (seperti gender, smoking_status, physical_activity, dietary_habits, air_pollution_exposure, stress_level, serta fitur biner lainnya) diubah menjadi representasi numerik menggunakan teknik One-Hot Encoding.
- Parameter drop='first' digunakan selama proses encoding ini untuk menghindari multicollinearity (dummy variable trap) dengan menghapus satu kolom dari setiap set kolom biner yang dihasilkan per fitur.
- Hasil dari proses ini adalah DataFrame encoded_cat_df yang berisi fitur-fitur kategorikal dalam format numerik (0 atau 1), dengan nama kolom yang dihasilkan secara otomatis (misalnya, gender_Male, smoking_status_Never, dll.). Indeks dari encoded_cat_df disamakan dengan df_no_outliers untuk memastikan konsistensi saat penggabungan.

**Pembentukan Matriks Fitur (X_raw) dan Vektor Target (y)**
Setelah fitur kategorikal di-encode:
   - Matriks fitur lengkap (X_raw) dibentuk dengan menggabungkan fitur-fitur numerik (yang diambil dari df_no_outliers[numeric_features]) dengan DataFrame hasil one-hot encoding (encoded_cat_df). Penggabungan dilakukan secara horizontal (axis=1).
   - Vektor target (y) disiapkan dengan mengambil kolom heart_attack dari df_no_outliers dan mengubah tipe datanya menjadi integer.
Langkah ini memastikan bahwa X_raw berisi semua fitur independen dalam format numerik, dan y berisi variabel dependen, keduanya dengan jumlah sampel yang konsisten.



**Pembagian Data (Train-Test Split)** : 
Dataset yang telah disiapkan (X_raw dan y) kemudian dibagi menjadi dua set: data latih (training set) dan data uji (test set).

- Proporsi pembagian adalah 80% untuk data latih dan 20% untuk data uji (test_size=0.2).
- Parameter random_state=42 digunakan untuk memastikan bahwa pembagian data bersifat reproduktif (hasilnya akan selalu sama setiap kali kode dijalankan).
- Parameter stratify=y digunakan untuk menjaga agar proporsi kelas target (jumlah kasus serangan jantung dan tidak serangan jantung) seimbang dan representatif pada kedua set data (latih dan uji) seperti pada dataset keseluruhan.

**Standardisasi Fitur (Feature Scaling)**
Fitur-fitur numerik dalam data latih dan data uji kemudian distandardisasi menggunakan StandardScaler dari scikit-learn.

- Objek StandardScaler diinisialisasi dan kemudian metode fit_transform() diterapkan pada data latih (X_train_raw). Proses fit() menghitung rata-rata dan standar deviasi dari data latih, dan transform() menerapkannya untuk menstandardisasi data latih tersebut menjadi X_train_scaled.
- Selanjutnya, metode transform() (tanpa fit() lagi) diterapkan pada data uji (X_test_raw) menggunakan parameter (rata-rata dan standar deviasi) yang telah dipelajari dari data latih. Hasilnya adalah X_test_scaled.
- Standardisasi ini penting karena mengubah skala fitur agar memiliki rata-rata 0 dan standar deviasi 1. Ini membantu algoritma yang sensitif terhadap skala fitur (seperti SVM, Logistic Regression dengan regularisasi, KNN) dan juga merupakan prasyarat yang baik untuk Principal Component Analysis (PCA).


**Principal Component Analysis (PCA) untuk Reduksi Dimensi** :
Setelah standardisasi, PCA diterapkan pada data latih yang telah diskalakan (X_train_scaled) untuk mengurangi jumlah dimensi (fitur) sambil berusaha mempertahankan sebanyak mungkin informasi (varians) penting dari data asli.

- Penentuan Jumlah Komponen Optimal: Jumlah komponen utama yang optimal ditentukan dengan menganalisis plot kumulatif explained variance ratio (seperti yang ditunjukkan pada gambar image_287db5.png atau image_33d2be.png). Berdasarkan analisis ini, 21 komponen utama (n_components_optimal = 21) dipilih karena mampu menjelaskan sekitar 93.58% dari total varians dalam data.
- Transformasi Data: Objek PCA dengan 21 komponen kemudian di-fit_transform pada X_train_scaled untuk menghasilkan X_train_pca_optimal, dan di-transform pada X_test_scaled untuk menghasilkan X_test_pca_optimal.
- Hasil Reduksi: Setelah PCA, data latih (X_train_pca_optimal) memiliki shape (120539, 21) dan data uji (X_test_pca_optimal) memiliki shape (30135, 21).

Dengan langkah-langkah persiapan data ini, dataset kini siap untuk digunakan dalam tahap pengembangan dan pelatihan model machine learning. Representasi fitur yang telah direduksi dimensinya namun tetap kaya informasi diharapkan dapat membantu model untuk belajar pola dengan lebih efektif.
## Modeling
Pada tahap ini, beberapa algoritma machine learning untuk klasifikasi diterapkan dan dilatih menggunakan data yang telah dipersiapkan secara optimal, yaitu X_train_pca_optimal (data latih yang telah melalui scaling dan reduksi dimensi menjadi 21 komponen utama menggunakan PCA) dan y_train (variabel target untuk data latih).

1. Logistic Regression:
- Cara Kerja: Logistic Regression adalah algoritma klasifikasi linier yang memprediksi probabilitas suatu kejadian dengan mencocokkan data pada fungsi logit. Model ini mencari hubungan linier antara fitur input dan log-odds dari kelas target.
- Implementasi & Parameter yang Digunakan:
   - Model diinisialisasi menggunakan LogisticRegression().
   - class_weight='balanced': Parameter ini digunakan untuk menyesuaikan bobot kelas secara otomatis guna menangani ketidakseimbangan kelas dalam variabel target heart_attack.
   - max_iter disesuaikan (misalnya, 500 atau 1000) untuk memastikan konvergensi.
   - Parameter lain seperti solver dan C (parameter regularisasi) menggunakan nilai default dari library scikit-learn pada iterasi ini.

2. Support Vector Machine (LinearSVC):
- Cara Kerja: Linear Support Vector Classification (LinearSVC) bertujuan untuk menemukan hyperplane terbaik yang memisahkan dua kelas dalam ruang fitur. Model ini efisien untuk data berdimensi tinggi dan dataset besar ketika menggunakan kernel linear.
- Implementasi & Parameter yang Digunakan:
   - Model diinisialisasi menggunakan LinearSVC(), yang merupakan implementasi SVM linear yang lebih cepat dibandingkan SVC(kernel='linear') untuk dataset besar.
   - Parameter yang digunakan antara lain random_state=42 untuk reproduktifitas, C=1.0 (parameter regularisasi default), max_iter=2000 (untuk memastikan konvergensi), dan dual=False (direkomendasikan ketika jumlah sampel lebih besar dari jumlah fitur).

3. K-Nearest Neighbors (KNN):
- Cara Kerja: KNN adalah algoritma non-parametrik berbasis jarak. Untuk mengklasifikasikan data baru, KNN mencari K sampel terdekat (tetangga) dari data tersebut dalam data latih. Kelas mayoritas di antara K tetangga tersebut kemudian ditetapkan sebagai kelas prediksi.
- Implementasi & Parameter yang Digunakan:
   - Model diinisialisasi menggunakan KNeighborsClassifier() dengan parameter default dari library scikit-learn. Ini umumnya termasuk n_neighbors=5 (jumlah tetangga), weights='uniform', dan metric='minkowski' (setara dengan jarak Euclidean).

4. XGBoost Classifier (Extreme Gradient Boosting):
- Cara Kerja: XGBoost adalah implementasi dari algoritma gradient boosting yang membangun model secara ensemble dari banyak decision tree secara sekuensial, di mana setiap tree baru memperbaiki kesalahan dari tree sebelumnya. XGBoost dikenal karena efisiensi dan performa tingginya.
- Implementasi & Parameter yang Digunakan:
   - Model diinisialisasi menggunakan XGBClassifier().
   - Parameter yang digunakan termasuk use_label_encoder=False (untuk menghindari peringatan), eval_metric='mlogloss', dan random_state=42.
   - Parameter fundamental lainnya seperti n_estimators, learning_rate, dan max_depth menggunakan nilai default dari library XGBoost pada iterasi ini.

Semua model dilatih menggunakan data X_train_pca_optimal dan y_train. Prediksi kemudian dilakukan pada X_test_pca_optimal untuk evaluasi performa.

## Evaluation
ahap evaluasi bertujuan untuk mengukur seberapa baik performa model-model yang telah dilatih dalam memprediksi serangan jantung pada data uji (X_test_pca_optimal dan y_test). Penggunaan data uji memastikan bahwa evaluasi dilakukan pada data yang belum pernah dilihat sebelumnya oleh model, sehingga memberikan gambaran yang lebih objektif tentang kemampuan generalisasi model.

### Metrik yang Digunakan

- Accuracy: Proporsi total prediksi yang benar dari keseluruhan data uji. Formula: (TP+TN)/(TP+TN+FP+FN)
- Precision (untuk kelas 1 - serangan jantung): Dari semua prediksi yang menyatakan pasien mengalami serangan jantung, berapa banyak yang benar-benar mengalami serangan jantung. Penting untuk menghindari false positive yang berlebihan. Formula: TP/(TP+FP)
- Recall (Sensitivity) (untuk kelas 1 - serangan jantung): Dari semua pasien yang sebenarnya mengalami serangan jantung, berapa banyak yang berhasil diprediksi dengan benar oleh model. Ini sangat krusial dalam konteks medis karena kegagalan mendeteksi kasus positif (false negative) bisa berakibat fatal. Formula: TP/(TP+FN)
- F1-score (untuk kelas 1 - serangan jantung): Rata-rata harmonik dari precision dan recall. Memberikan keseimbangan antara kedua metrik tersebut. Formula: 2×(Precision×Recall)/(Precision+Recall)

**Dimana:**
- TP (True Positive): Pasien serangan jantung yang diprediksi serangan jantung.
- TN (True Negative): Pasien tidak serangan jantung yang diprediksi tidak serangan jantung.
- FP (False Positive): Pasien tidak serangan jantung yang diprediksi serangan jantung.
- FN (False Negative): Pasien serangan jantung yang diprediksi tidak serangan jantung.


### Hasil Evaluasi Model

| Model               | Accuracy | Precision (1) | Recall (1) | F1-score (1) |
|---------------------|----------|----------------|------------|--------------|
| Logistic Regression | 0.7020	| 0.61           | 0.69       | 0.65         |
| SVM (Linear SVC)    | 0.7172   | 0.68           | 0.54       | 0.60         |
| KNN                 | 0.6861   | 0.63           | 0.52       | 0.57         |
| XGBoost             | 0.7216   | 0.67           | 0.59       | 0.63         |

### Analisis Hasil dan Hubungan dengan Business Understanding:
Setelah melakukan perbaikan pada tahap persiapan data, khususnya dengan menggunakan 21 komponen utama hasil PCA yang menangkap ~93.6% varians data, serta memastikan evaluasi dilakukan pada data uji, diperoleh hasil performa model yang lebih realistis.

**Menjawab Problem Statement (PS1 & PS2):**
- **PS1 (Bagaimana cara memprediksi kemungkinan serangan jantung?):** Proyek ini menunjukkan bahwa machine learning, setelah melalui persiapan data yang cermat termasuk reduksi dimensi dengan PCA dan penskalaan, dapat digunakan untuk memprediksi kemungkinan serangan jantung. Keempat model yang diuji memberikan tingkat akurasi antara 68% hingga 72% pada data uji.
- **PS2 (Algoritma machine learning mana yang paling efektif?):**
   - Berdasarkan akurasi keseluruhan tertinggi, XGBoost (72.16%) menunjukkan performa sedikit lebih unggul, diikuti oleh SVM (LinearSVC) (71.72%).
   - Namun, jika fokus pada kemampuan mendeteksi kasus serangan jantung (recall kelas 1), Logistic Regression dengan class_weight='balanced' (recall 0.69) adalah yang terbaik di antara model yang diuji. Ini berarti Logistic Regression paling banyak berhasil mengidentifikasi pasien yang benar-benar mengalami serangan jantung.
   - Untuk precision kelas 1 (keakuratan prediksi positif), SVM (LinearSVC) (0.68) dan XGBoost (0.67) menunjukkan hasil terbaik.
   - Tidak ada satu model yang unggul mutlak di semua metrik. Pemilihan model "paling efektif" akan bergantung pada prioritas metrik mana yang dianggap paling penting untuk kasus penggunaan ini (misalnya, meminimalkan false negative mungkin lebih krusial).

- **Mencapai Goals (G1 & G2):**
   - G1 (Membangun model prediksi dengan akurasi dan performa metrik yang layak): Dengan akurasi tertinggi mencapai ~72% dan recall untuk kasus serangan jantung mencapai ~69% (oleh Logistic Regression), model-model ini menunjukkan potensi. Apakah ini "layak" untuk implementasi klinis masih memerlukan pertimbangan lebih lanjut dan standar yang lebih tinggi, terutama untuk recall. Namun, ini adalah peningkatan signifikan dari iterasi sebelumnya.
   - G2 (Membandingkan performa beberapa algoritma): Tujuan ini tercapai. Empat algoritma telah diimplementasikan dan dievaluasi pada representasi data yang sama (21 komponen PCA), memberikan perbandingan yang adil dan wawasan tentang trade-off masing-masing.

- **Dampak Solution Statements:**
   - Menerapkan empat algoritma klasifikasi: Memungkinkan perbandingan langsung dan identifikasi kekuatan relatif masing-masing model pada dataset yang telah diproses.
   - Melakukan balancing pada kelas target (class_weight='balanced' untuk Logistic Regression): Terbukti efektif meningkatkan recall untuk kelas minoritas (serangan jantung) pada model Logistic Regression, menjadikannya kandidat kuat jika prioritasnya adalah sensitivitas terhadap kasus positif.
   - Mengevaluasi performa dengan metrik komprehensif: Penggunaan accuracy, precision, recall, dan F1-score sangat penting untuk memahami performa model secara menyeluruh, terutama trade-off antara mendeteksi kasus positif dan keakuratan prediksi positif tersebut.

### Kesimpulan Evaluasi:
Setelah optimasi pada tahap persiapan data dengan menggunakan 21 komponen PCA, semua model menunjukkan peningkatan performa yang signifikan dibandingkan dengan penggunaan 2 komponen PCA. XGBoost mencapai akurasi tertinggi secara keseluruhan (72.16%) pada data uji. Namun, Logistic Regression (dengan class_weight='balanced') menunjukkan recall tertinggi untuk kasus serangan jantung (kelas 1) sebesar 0.69, yang merupakan metrik krusial dalam konteks medis untuk meminimalkan kasus yang terlewat (false negative). SVM dalam bentuk LinearSVC menunjukkan presisi tertinggi untuk kelas 1 (0.68) dan waktu pelatihan yang sangat cepat. KNN, meskipun meningkat, menunjukkan recall terendah untuk kelas 1 di antara model yang lebih baik.

Pilihan model terbaik akan bergantung pada metrik mana yang diprioritaskan. Jika tujuannya adalah untuk memaksimalkan deteksi kasus serangan jantung, Logistic Regression adalah pilihan yang menonjol. Jika akurasi keseluruhan atau presisi prediksi positif sedikit lebih diutamakan, XGBoost atau LinearSVC bisa dipertimbangkan.

Meskipun ada peningkatan, performa model (terutama recall untuk kelas 1 yang belum mencapai tingkat sangat tinggi) menunjukkan bahwa masih ada ruang untuk perbaikan lebih lanjut. Langkah-langkah seperti hyperparameter tuning yang lebih ekstensif untuk semua model (terutama XGBoost dan LinearSVC), eksplorasi penggunaan fitur tanpa PCA (hanya dengan scaling), atau teknik penanganan ketidakseimbangan data yang lebih lanjut seperti SMOTE (diterapkan hanya pada data latih) dapat dipertimbangkan untuk iterasi berikutnya guna meningkatkan performa model secara keseluruhan.


