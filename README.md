# ProjectML_261-235
# Klasifikasi Covid-19 Menggunakan Citra X-Ray Radiology

![alt text](https://miro.medium.com/max/2625/1*YrvMKrWMhi3HomoiTLPsfw.png) ![alt](https://keras.io/img/logo.png)

## Deskripsi Dataset 

Dalam penyusunan project ini menggunakan 2 dataset dari sumber yang berbeda yang nantinya akan dikombinasikan; 
Dataset pertama berasal dari penelitian Tulin Otzurk, dkk yang berjudul [Automated detection of COVID-19 cases using deep neural networks with X-ray images](https://www.sciencedirect.com/science/article/abs/pii/S0010482520301621?via%3Dihub) pada tahun 2020
Dataset memiliki jumlah total sebanyak 1125 image yang terdiri atas 3 kelas dengan rincian :
1. Covid-19     : 125 images
2. No Findings  : 500 images
3. Pneumonia    : 500 images
dataset : https://github.com/drkhan107/CoroNet

Kemudian, dataset kedua yang digunakan dalam pembuatan project ini adalah COVID-19 Chest X-ray yang diakses melalui penelitian dari K. Ashif Iqbal, dkk [CoroNet: A deep neural network for detection and diagnosis of COVID-19 from chest x-ray images](https://www.sciencedirect.com/science/article/abs/pii/S0169260720314140?via%3Dihub) dengan tahun yang sama
Dataset terdiri atas 4 kelas dengan jumlah total 1638 image, dengan detail sebagai berikut : 
1. Covid                : 320 images 
2. Normal               : 445 images
3. Pneumonia Bacterial  : 449 images 
4. Pneumonia Viral      : 424 images
dataset : https://github.com/muhammedtalo/COVID-19

### Teknik Deep Learning yang digunakan

* Model dengan menggunakan algoritma ANN (Artificial Neural Network)
* Model dengan menggunakan algoritma CNN (Convolutional Neural Network)
* Model dengan menggunakan algoritma Transfer Learning Resnet50
* Model dengan menggunakan algoritma Transfer Learning DenseNet50
* Model dengan menggunakan algoritma InceptionV3
model dilatih dengan menggunakan google colabolatory. 

### Jurnal referensi 

* Jurnal referensi pada projek ini berjudul [Attention-based VGG-16 model for COVID-19 chest X-ray image classification](https://link.springer.com/article/10.1007/s10489-020-02055-x)
Sitaula, C., Hossain, M.B. Attention-based VGG-16 model for COVID-19 chest X-ray image classification. Appl Intell 51, 2850–2863 (2021). https://doi.org/10.1007/s10489-020-02055-x

### Cara Pengeksekusian Program

* Download file .ipynb 
* Buka di Notebook anda atau [google colabolatory](https://colab.research.google.com/) 

## Authors

Kontributor dalam projek ini yaitu :
* Ulfah Nur Oktaviana ulfanuroktaviana575@webmail.umm.ac.id
* Tiara Intana Sari tiaraintana@gmail.com

## Acknowledgments

Inspiration, code snippets, etc.
* [Covid19 Radiography Database](https://www.kaggle.com/tawsifurrahman/covid19-radiography-database)
* [CNN Tensorflow](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v1/examples/3_NeuralNetworks/convolutional_network.py)
* [DenseNet Documentation](https://keras.io/api/applications/densenet/)
* [Resnet Documentation](https://keras.io/api/applications/resnet/)
* [Keras Documentation of Transfer Learning Model](https://keras.io/api/applications/)
* [Data AUgmentation Using Tensorflow](https://www.tensorflow.org/tutorials/images/data_augmentation)

