# **Sleep Health Prediction System: XGBoost Model**

**Deskripsi Proyek:**

Proyek ini bertujuan untuk memprediksi gangguan tidur berdasarkan data gaya hidup menggunakan model XGBoost. Model ini dilatih menggunakan data kesehatan tidur, dan dilengkapi dengan Continuous Integration (CI) menggunakan GitHub Actions untuk otomatisasi pengujian, pembangunan, dan deployment model.

---

## **Fitur Utama**

* **Model XGBoost:** Menggunakan XGBoost untuk membangun model prediksi gangguan tidur.
* **CI/CD dengan GitHub Actions:** CI digunakan untuk otomatisasi pengujian dan deployment.
* **MLflow:** Digunakan untuk mengelola siklus hidup model, mulai dari pelatihan hingga deployment.

---

## **Struktur Proyek**

1. **MLProject/**: Folder yang berisi kode dan konfigurasi untuk pelatihan model.

   * `modelling.py`: Kode untuk pelatihan model menggunakan XGBoost.
   * `conda.yaml`: File yang mendefinisikan lingkungan Python yang dibutuhkan untuk proyek.
2. **.github/main.yml**: File berisi konfigurasi GitHub Actions untuk Continuous Integration.
3. **Dockerfile**: Jika ingin menggunakan Docker untuk deployment model.
   Link: [Docker Hub](https://hub.docker.com/r/labibaadinda/xgb_model_image)

---

## **Setup dan Instalasi**

### 1. Clone repositori ini ke mesin lokal Anda:

```bash
git clone https://github.com/labibaadinda/Workflow-CI
```

### 2. Buat dan aktifkan environment menggunakan Conda:

```bash
conda env create -f MLProject/conda.yaml
conda activate mlflow-env
```

### 3. Install dependensi yang diperlukan:

```bash
cd MLProject
pip install -r requirements.txt
```

---

## **Pelatihan Model**

Proyek ini menggunakan **MLflow** untuk melatih model dan menyimpan artefak pelatihan. Anda dapat melatih model menggunakan file `modelling.py`.

1. **Jalankan pelatihan model dengan perintah berikut:**

```bash
python MLProject/modelling.py
```

2. **Model akan dilatih menggunakan XGBoost dengan parameter yang telah ditentukan. Setelah pelatihan, model akan disimpan ke dalam direktori `xgboost_model_dir`.**

---

## **Workflow CI dengan GitHub Actions**

Proyek ini menggunakan **GitHub Actions** untuk otomatisasi pengujian, pembangunan, dan deployment model.

### Deskripsi Workflow:

1. **CI Pipeline Trigger:**

   * Workflow akan dijalankan setiap kali ada perubahan pada branch `main` atau pull request yang mengarah ke branch tersebut.

2. **Langkah-langkah Workflow:**

   * **Checkout repository:** Mengambil kode dari repositori GitHub atau Link [Docker Hub](https://hub.docker.com/r/labibaadinda/xgb_model_image)
 ini. 
   * **Set up Miniconda:** Menyiapkan environment dengan `conda.yaml` yang didefinisikan dalam folder `MLProject`.
   * **Pelatihan model menggunakan MLflow:** Model XGBoost akan dilatih dengan data yang ada dan hasilnya disimpan.
   * **Verifikasi model:** Memastikan bahwa file model yang telah dilatih ada sebelum di-upload.
   * **Upload model ke GitHub:** Menyimpan model sebagai artefak.
   * **Build Docker Image:** Membuat image Docker untuk model yang telah dilatih.
   * **Login dan Push Docker Image ke Docker Hub:** Login ke Docker Hub dan meng-upload image ke Docker Hub.

### File Workflow CI:

File konfigurasi workflow berada pada `Workflow-CI/main.yml`.

---

## **Penggunaan Docker**

1. **Build Docker image dari model yang telah dilatih:**

```bash
mlflow models build-docker -m "file://$(pwd)/xgboost_model_dir" --name xgb_model_image
```

2. **Login ke Docker Hub dan push image ke Docker Hub:**

```bash
docker login -u <your-docker-hub-username> -p <your-docker-hub-access-token>
docker tag xgb_model_image:latest <your-docker-hub-username>/xgb_model_image:latest
docker push <your-docker-hub-username>/xgb_model_image:latest
```

---


## **Catatan**

* Semua dependensi yang dibutuhkan untuk proyek ini tercatat dalam file `MLProject/conda.yaml` dan `requirements.txt`.
* Gunakan **MLflow** untuk melacak eksperimen dan menyimpan model yang telah dilatih.
* CI/CD pipeline akan otomatis dijalankan untuk memastikan bahwa model berfungsi dengan baik pada setiap perubahan kode.

