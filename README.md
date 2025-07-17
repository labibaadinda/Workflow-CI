### Sleep Disorder Prediction using XGBoost 

## **Project Overview**

Proyek ini bertujuan untuk membangun pipeline **Machine Learning** untuk **prediksi sleep disorder** menggunakan **XGBoost** dengan integrasi **MLflow** untuk manajemen eksperimen dan **GitHub Actions** untuk **CI/CD**. Model yang dihasilkan kemudian akan dibangun menjadi **Docker Image** dan dipublikasikan ke **Docker Hub** untuk kemudahan deployment dan integrasi.

---

## **Alur Kerja Proyek**

### 1. **Pengaturan Lingkungan (Environment Setup)**

* **Tujuan**: Menyiapkan lingkungan untuk proyek ini menggunakan Conda.
* **Deskripsi**:

  * Menggunakan file **`conda.yaml`** untuk membuat lingkungan Conda dengan dependensi yang diperlukan seperti `mlflow`, `xgboost`, `scikit-learn`, dan lainnya.
  * Anda dapat menginstal environment menggunakan perintah:

    ```bash
    conda env create -f conda.yaml
    conda activate mlflow-enva
    ```

### 2. **Data Preparation**

* **Tujuan**: Menyiapkan dataset untuk pelatihan model.
* **Deskripsi**:

  * Dataset yang digunakan adalah **`sleep-health_life-style_preprocessing.csv`** yang berisi data yang digunakan untuk memprediksi gangguan tidur.
  * Kolom target yang diprediksi adalah **`Sleep Disorder`**.

### 3. **Pengembangan Model (Model Development)**

* **Tujuan**: Membangun dan melatih model **XGBoost** untuk prediksi gangguan tidur.
* **Deskripsi**:

  * **`modelling.py`** berfungsi untuk melatih model menggunakan **XGBoost** dan melakukan hyperparameter tuning menggunakan **GridSearchCV**.
  * Model yang dilatih akan disimpan menggunakan **MLflow** dan file backup model disimpan dengan **Joblib**.

  Langkah-langkah dalam **`modelling.py`**:

  * Membaca dataset dan memisahkan fitur serta target.
  * Melakukan pelatihan model dengan berbagai kombinasi hyperparameter.
  * Menyimpan hasil eksperimen dan model yang terlatih di MLflow.
  * Mengevaluasi performa model dengan metrik seperti akurasi, precision, recall, dan F1-score.

  ```bash
  python modelling.py
  ```

### 4. **Manajemen Eksperimen dengan MLflow**

* **Tujuan**: Melacak dan mencatat hasil eksperimen menggunakan MLflow.
* **Deskripsi**:

  * Selama pelatihan, **MLflow** digunakan untuk:

    * Mencatat metrik performa model seperti **accuracy**, **precision**, **recall**, dan **f1-score**.
    * Menyimpan model terbaik di dalam direktori **MLflow model**.
    * Mencatat artefak seperti file model menggunakan **Joblib**.

  **Contoh Penggunaan**:

  * Semua eksperimen akan tercatat di **MLflow** untuk memastikan pelatihan model dapat diulang dan divalidasi kapan saja.

### 5. **Automatisasi Pelatihan dan Deploy Menggunakan GitHub Actions**

* **Tujuan**: Mengotomatiskan pelatihan model dengan CI/CD menggunakan **GitHub Actions**.
* **Deskripsi**:

  * **GitHub Actions** digunakan untuk memicu workflow pelatihan model ketika ada perubahan pada branch `main` atau saat pull request dibuat.
  * Workflow akan melakukan:

    1. Mengatur environment Conda.
    2. Menjalankan pelatihan model menggunakan **MLflow**.
    3. Verifikasi model yang dilatih.
    4. Meng-upload model yang dilatih ke GitHub sebagai artefak.
    5. Membangun Docker image dari model yang dilatih.



### 6. **Pembuatan Docker Image dan Deployment**

* **Tujuan**: Membangun dan mendorong Docker image ke Docker Hub.
* **Deskripsi**:

  * Setelah model dilatih, **MLflow** digunakan untuk membangun **Docker image** yang berisi model.
  * Docker image kemudian didorong ke **Docker Hub** untuk deployment lebih lanjut.
  * Langkah-langkah di **GitHub Actions** mengotomatisasi proses ini.

  **Docker Build Example**:

  ```bash
  mlflow models build-docker -m "file://$(pwd)/xgboost_model_dir" --name xgb_model_image
  ```

### 7. **Verifikasi dan Pengujian**

* **Tujuan**: Memastikan bahwa seluruh pipeline berfungsi dengan benar.
* **Deskripsi**:

  * Setelah pipeline berjalan, pastikan model yang dilatih telah disimpan dengan benar dan Docker image dapat di build dan di push ke Docker Hub tanpa error.


---

## **Struktur Direktori**

```plaintext
WORKFLOW-CI
├── MLProject/
│   ├── conda.yaml                        # File environment untuk Conda
│   ├── dockerhub.txt                     # File kredensial untuk Docker Hub
│   ├── MLproject                         # File konfigurasi MLflow Project
│   ├── modelling.py                      # Script pelatihan model
│   ├── requirements.txt                  # Daftar dependensi Python
│   ├── sleep-health_life-style_preprocessing.csv  # Dataset
│   └── xgboost_best_model.joblib         # Model XGBoost terbaik yang disimpan
├── .github/
│   └── workflows/
│       └── main.yml                      # GitHub Actions CI/CD pipeline
├── mlruns/                               # Direktori MLflow untuk menyimpan artefak eksperimen
├── xgboost_model_dir                     # Direktori model yang disimpan oleh MLflow
└── README.md                             # Dokumentasi proyek

```

---

### **Bagaimana Cara Menggunakan Proyek Ini?**

1. **Install Dependencies**:

   * Gunakan Conda untuk membuat environment:

     ```bash
     conda env create -f conda.yaml
     conda activate mlflow-enva
     ```

2. **Jalankan Pelatihan Model**:

   * Anda bisa menjalankan pelatihan model secara manual menggunakan:

     ```bash
     python modelling.py
     ```

3. **Automatisasi dengan GitHub Actions**:

   * Pastikan GitHub Actions diatur untuk memicu pelatihan model pada push ke branch `main`.

4. **Buat dan Deploy Docker Image**:

   * Setelah model dilatih, Docker image dibuild dan push ke Docker Hub secara otomatis.

