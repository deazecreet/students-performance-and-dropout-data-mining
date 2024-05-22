# Proyek Akhir: Menyelesaikan Permasalahan Institusi Pendidikan menggunakan *Workflow* **CRISP-DM**

## Business Understanding
Jaya Jaya Institut adalah sebuah institusi pendidikan yang telah berdiri sejak tahun 2000 yang telah banyak mencetak lulusan berprestasi dengan reputasi yang gemilang. Namun, di balik semua kesuksesan itu, ada satu masalah besar yang terus menghantui, yaitu tingginya angka *dropout* atau mahasiswa yang tidak menyelesaikan pendidikannya. Masalah ini tidak hanya berdampak pada reputasi institusi, tetapi juga merugikan para mahasiswa yang kehilangan kesempatan untuk meraih masa depan yang lebih baik.

### Permasalahan Bisnis
1. **Tingginya Angka Dropout**: Banyak mahasiswa yang tidak menyelesaikan pendidikan mereka. Hal ini bisa disebabkan oleh berbagai faktor seperti masalah akademis, finansial, atau personal.

2. **Minimnya Deteksi Dini**: Saat ini, institusi belum memiliki sistem yang efektif untuk mendeteksi mahasiswa yang berisiko dropout sejak dini. Tanpa deteksi dini, sulit untuk memberikan intervensi yang tepat waktu dan efektif.

3. **Kurangnya Intervensi yang Tepat**: Karena tidak ada sistem deteksi dini, intervensi yang diberikan seringkali terlambat atau tidak sesuai dengan kebutuhan mahasiswa, sehingga tidak berhasil menurunkan angka dropout.

### Cakupan Proyek
1. *Business Understanding*: Mengidentifikasi tujuan proyek, menentukan pertanyaan bisnis yang harus dijawab, dan menyusun rencana proyek.

2. *Data Understanding*: 

    * Mengumpulkan data mahasiswa yang tersedia, termasuk data demografis, akademis, dan faktor-faktor lainnya. 

    * Melakukan analisis awal untuk memahami struktur data, mengidentifikasi pola dan distribusi, serta menemukan anomali dan *outliers*.

    * Mengecek kualitas data dengan mengidentifikasi *missing values*, data yang tidak valid, dan inkonsistensi.

3. *Data Preparation*

    * Membersihkan data dengan menangani *missing values*, mengoreksi data yang tidak valid, dan mengatasi inkonsistensi.

    * Melakukan transformasi data yang diperlukan, seperti *encoding* variabel kategorikal dan normalisasi fitur numerik.

    * Menyusun dataset yang akan digunakan untuk pemodelan dengan memilih fitur-fitur yang relevan dan membagi data menjadi set pelatihan dan pengujian.

4. *Modeling*

    * Memilih algoritma *machine learning* yang sesuai untuk masalah prediksi *dropout*, seperti *Logistic Regression*, *Decision Tree*, *Random Forest*, dan *Gradient Boosting*.

    * Mengevaluasi performa masing-masing model menggunakan data pengujian dengan metrik seperti *accuracy*, *precision*, *recall*, dan *F1 score*.

5. *Evaluation*

    * Menilai hasil dari model yang telah dibangun untuk memastikan bahwa model memenuhi tujuan bisnis dan memberikan prediksi yang akurat.

6. *Deployment*

    * Mengimplementasikan model terbaik ke dalam bentuk *prototype* yang dibuat menggunakan Streamlit untuk sistem deteksi dini yang dapat digunakan oleh Jaya Jaya Institut untuk mendeteksi mahasiswa yang berisiko tinggi *dropout*.

    * Mengembangkan *dashboard* interaktif menggunakan Tableau untuk membantu Jaya Jaya Institut dalam memantau performa mahasiswa, memahami data demografis dan akademis, serta memonitor *dropout*.

### Persiapan

Sumber data: [Link Dataset](https://github.com/dicodingacademy/dicoding_dataset/blob/main/students_performance/data.csv)

Setup environment:
```
pip install pandas matplotlib seaborn numpy scikit-learn joblib factor-analyzer streamlit
```

## Business Dashboard
*Business Dashboard* dibuat berdasarkan variabel-variabel yang memiliki pola menarik yang diketahui pada step *Data Understanding*.

[Kunjungi Dashboard](https://public.tableau.com/views/DashboardAnalisisDropoutdanKinerjaMahasiswa/Story1?:language=en-US&:sid=&:display_count=n&:origin=viz_share_link)

## Menjalankan Sistem Machine Learning
Untuk menjalankan *prototype* sistem *machine learning*, letakkan folder `submission` di folder proyek Anda, asalkan bukan open folder submission. Lalu, jalankan perintah di bawah pada Terminal.

```
streamlit run submission/app.py
```

[Kunjungi Prototype Aplikasi](https://students-performance-dataset-analytics-azelrizkinasution.streamlit.app/)

## Conclusion
Model *Gradient Boosting* yang saya buat berhasil memberikan prediksi yang cukup baik dalam mendeteksi mahasiswa yang kemungkinan besar akan *dropout*. Faktor terpenting dalam memonitor performa mahasiswa di Jaya Jaya Institut adalah faktor `PC1`, yang mana faktor tersebut sangat dipengaruhi oleh variabel jumlah unit kurikuler yang diambil oleh mahasiswa pada semester pertama. Hal tersebut menunjukkan bahwa mahasiswa yang mengambil lebih banyak unit kurikuler pada semester pertama cenderung memiliki performa akademik yang lebih baik dan kemungkinan lebih kecil untuk *dropout*.

### Rekomendasi Action Items
1. **Peningkatan Dukungan Akademis**: Menyediakan layanan konseling akademis, tutor, dan mentor untuk membantu mahasiswa dalam mengelola beban studi mereka, khususnya pada semester pertama yang krusial.

2. **Penyesuaian Kurikulum dan Beban Studi**: Meninjau dan menyesuaikan kurikulum agar lebih fleksibel dan sesuai dengan kemampuan serta kebutuhan mahasiswa dengan mempertimbangkan beban studi yang optimal untuk mendukung kesuksesan akademis mereka.

3. **Pemantauan Berkelanjutan dan Evaluasi**: Melakukan pemantauan berkelanjutan terhadap performa akademik mahasiswa dan secara berkala mengevaluasi efektivitas intervensi yang telah dilakukan dengan tujuan untuk terus meningkatkan strategi dukungan akademis.
