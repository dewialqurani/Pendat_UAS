import streamlit as st

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import neighbors, datasets
from sklearn.preprocessing import MinMaxScaler

import pickle

from sklearn import metrics

st.set_page_config(
    page_title="Prediksi Penyakit Jantung"
)

st.title('Prediksi Penyakit jantung')
st.write("""
Aplikasi Untuk Memprediksi Kemungkinan Penyakit Jantung
""")

tab1, tab2, tab3, tab4 = st.tabs(["Data Understanding", "Preprocecing", "Modelling", "Implementation"])

with tab1:
    st.write("""
    <h5>Data Understanding</h5>
    <br>
    """, unsafe_allow_html=True)

    st.markdown("""
    Link Dataset
    <a href="https://www.kaggle.com/datasets/cherngs/heart-disease-cleveland-uci"> https://www.kaggle.com/datasets/cherngs/heart-disease-cleveland-uci</a>
    """, unsafe_allow_html=True)

    st.markdown("""
    Link Repository Github

    """, unsafe_allow_html=True)

    df = pd.read_csv("heart.csv")
    st.write("Dataset Heart Disease : ")
    st.write(df)
    st.write("Penjelasan kolom-kolom yang ada")

    st.write("""
    <ol>
    <li>age : Umur dalam satuan Tahun</li>
    <li>sex : Jenis Kelamin (1=Laki-laki, 0=Perempuan)</li>
    <li>cp : chest pain type (tipe sakit dada)(0 = typical angina, 1 = atypical angina, 2 = non-anginal pain, 3 = asymptomatic)</li>
    <li>trestbps : tekanan darah saat dalam kondisi istirahat dalam mm/Hg</li>
    <li>chol : serum sholestoral (kolestrol dalam darah) dalam Mg/dl </li>
    <li>fbs : fasting blood sugar (kadar gula dalam darah setelah berpuasa) lebih dari 120 mg/dl (1=Iya, 0=Tidak)</li>
    <li>restecg : hasil test electrocardiographic (0 = normal, 1 = memiliki kelainan gelombang ST-T (gelombang T inversi dan/atau ST elevasi atau depresi > 0,05 mV), 2 = menunjukkan kemungkinan atau pasti hipertrofi ventrikel kiri dengan kriteria Estes)</li>
    <li>thalach : rata-rata detak jantung pasien dalam satu menit</li>
    <li>exang :  keadaan dimana pasien akan mengalami nyeri dada apabila berolah raga, 0 jika tidak nyeri, dan 1 jika menyebabkan nyeri</li>
    <li>oldpeak : depresi ST yang diakibatkan oleh latihan relative terhadap saat istirahat</li>
    <li>Slope : slope dari puncak ST setelah berolah raga. Atribut ini memiliki 3 nilai yaitu 0 untuk downsloping, 1 untuk flat, dan 2 untuk upsloping.</li>
    <li>Ca: banyaknya pembuluh darah yang terdeteksi melalui proses pewarnaan flourosopy</li>
    <li>Thal: detak jantung pasien. Atribut ini memiliki 3 nilai yaitu 1 untuk fixed defect, 2 untuk normal dan 3 untuk reversal defect</li>
    <li>target: hasil diagnosa penyakit jantung, 0 untuk terdiagnosa positif terkena penyakit jantung koroner, dan 1 untuk negatif terkena penyakit jantung koroner.</li>
    </ol>
    """,unsafe_allow_html=True)

with tab2:
    st.write("""
    <h5>Preprocessing Data</h5>
    <br>
    """, unsafe_allow_html=True)
    st.write("""
    <p style="text-align: justify;text-indent: 45px;">Preprocessing data adalah proses mengubah data mentah ke dalam bentuk yang lebih mudah dipahami. Proses ini diperlukan untuk memperbaiki kesalahan pada data mentah yang seringkali tidak lengkap dan memiliki format yang tidak teratur. Preprocessing melibatkan proses validasi dan imputasi data.</p>
    <p style="text-align: justify;text-indent: 45px;">Salah satu tahap Preprocessing data adalah Normalisasi. Normalisasi data adalah elemen dasar data mining untuk memastikan record pada dataset tetap konsisten. Dalam proses normalisasi diperlukan transformasi data atau mengubah data asli menjadi format yang memungkinkan pemrosesan data yang efisien.</p>
    <br>
    """,unsafe_allow_html=True)
    scaler = st.radio(
    "Pilih metode normalisasi data",
    ('Tanpa Scaler', 'MinMax Scaler'))
    if scaler == 'Tanpa Scaler':
        st.write("Dataset Tanpa Preprocessing : ")
        df_new=df
    elif scaler == 'MinMax Scaler':
        st.write("Dataset setelah Preprocessing dengan MinMax Scaler: ")
        scaler = MinMaxScaler()
        df_for_scaler = pd.DataFrame(df, columns = ['age','trestbps','chol','thalach','oldpeak'])
        df_for_scaler = scaler.fit_transform(df_for_scaler)
        df_for_scaler = pd.DataFrame(df_for_scaler,columns = ['age','trestbps','chol','thalach','oldpeak'])
        df_drop_column_for_minmaxscaler=df.drop(['age','trestbps','chol','thalach','oldpeak'], axis=1)
        df_new = pd.concat([df_for_scaler,df_drop_column_for_minmaxscaler], axis=1)
    st.write(df_new)

with tab3:
    st.write("""
    <h5>Modelling</h5>
    <br>
    """, unsafe_allow_html=True)

    nb = st.checkbox("Naive bayes") #chechbox naive bayes
    knn = st.checkbox("KNN")
    ds = st.checkbox("Decission Tree")


with tab4:
    st.write("""
    <h5>Implementation</h5>
    <br>
    """, unsafe_allow_html=True)
    X=df_new.iloc[:,0:13].values
    y=df_new.iloc[:,13].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,stratify=y, random_state=42)

    age=st.number_input("umur : ")
    sex=st.selectbox(
        'Pilih Jenis Kelamin',
        ('Laki-laki','Perempuan')
    )
    if sex=='Laki-laki':
        sex=1
    elif sex=='Perempuan':
        sex=0
    cp=st.selectbox(
        'Jenis nyeri dada',
        ('Typical Angina','Atypical angina','non-anginal pain','asymptomatic')
    )
    if cp=='Typical Angina':
        cp=0
    elif cp=='Atypical angina':
        cp=1
    elif cp=='non-anginal pain':
        cp=2
    elif cp=='asymptomatic':
        cp=3
    trestbps=st.number_input('resting blood pressure / tekanan darah saat kondisi istirahat(mm/Hg)')
    chol=st.number_input('serum cholestoral / kolestrol dalam darah (Mg/dl)')
    fbs=st.selectbox(
        'fasting blood sugar / gula darah puasa',
        ('Dibawah 120', 'Diatas 120')
    )
    if fbs=='Dibawah 120':
        fbs=0
    elif fbs=='Diatas 120':
        fbs=1
    restecg=st.selectbox(
        'resting electrocardiographic results',
        ('normal','mengalami kelainan gelombang ST-T','menunjukkan kemungkinan atau pasti hipertrofi ventrikel kiri dengan kriteria Estes')    
    )
    if restecg=='normal':
        restecg=0
    elif restecg=='mengalami kelainan gelombang ST-T':
        restecg=1
    elif restecg=='menunjukkan kemungkinan atau pasti hipertrofi ventrikel kiri dengan kriteria Estes':
        restecg=2
    thalach=st.number_input('thalach (rata-rata detak jantung pasien dalam satu menit)')
    exang=st.selectbox(
        'exang/exercise induced angina',
        ('ya','tidak')
    )
    if exang=='ya':
        exang=1
    elif exang=='tidak':
        exang=0
    oldpeak=st.number_input('oldpeak/depresi ST yang diakibatkan oleh latihan relative terhadap saat istirahat')
    slope=st.selectbox(
        'slope of the peak exercise',
        ('upsloping','flat','downsloping')
    )
    if slope=='upsloping':
        slope=0
    elif slope=='flat':
        slope=1
    elif slope=='downsloping':
        slope=2
    ca=st.number_input('number of major vessels')
    thal=st.selectbox(
        'Thalassemia',
        ('normal','cacat tetap','cacat reversibel')
    )
    if thal=='normal':
        thal=0
    elif thal=='cacat tetap':
        thal=1
    elif thal=='cacat reversibel':
        thal=2

    algoritma = st.selectbox(
        'pilih algoritma klasifikasi',
        ('KNN','Naive Bayes','Random Forest','Ensemble Stacking')
    )
    prediksi=st.button("Diagnosis")
    if prediksi:
        if algoritma=='KNN':
            model = KNeighborsClassifier(n_neighbors=3)
            filename='knn.pkl'
        elif algoritma=='Naive Bayes':
            model = GaussianNB()
            filename='gaussian.pkl'
        elif algoritma=='Random Forest':
            model = RandomForestClassifier(n_estimators = 100)
            filename='randomforest.pkl'
        elif algoritma=='Ensemble Stacking':
            estimators = [
                ('rf_1', RandomForestClassifier(n_estimators=10, random_state=42)),
                ('knn_1', KNeighborsClassifier(n_neighbors=10))             
            ]
            model = StackingClassifier(estimators=estimators, final_estimator=GaussianNB())
            filename='stacking.pkl'
        
        
        model.fit(X_train, y_train)
        Y_pred = model.predict(X_test) 

        score=metrics.accuracy_score(y_test,Y_pred)

        loaded_model = pickle.load(open(filename, 'rb'))
        if scaler == 'Tanpa Scaler':
            dataArray = [age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]
        else:
            age_proceced = (age - df['age'].min(axis=0)) / (df['age'].max(axis=0) - df['age'].min(axis=0))
            trestbps_proceced = (trestbps - df['trestbps'].min(axis=0)) / (df['trestbps'].max(axis=0) - df['trestbps'].min(axis=0))
            chol_proceced = (chol - df['chol'].min(axis=0)) / (df['chol'].max(axis=0) - df['chol'].min(axis=0))
            thalach_proceced = (thalach - df['thalach'].min(axis=0)) / (df['thalach'].max(axis=0) - df['thalach'].min(axis=0))
            oldpeak_proceced = (oldpeak - df['oldpeak'].min(axis=0)) / (df['oldpeak'].max(axis=0) - df['oldpeak'].min(axis=0))
            dataArray = [age_proceced, trestbps_proceced, chol_proceced, thalach_proceced, oldpeak_proceced, sex, cp, fbs, restecg, exang, slope, ca, thal]
        pred = loaded_model.predict([dataArray])

        if int(pred[0])==0:
            st.success(f"Hasil Prediksi : Tidak memiliki penyakit Jantung")
        elif int(pred[0])==1:
            st.error(f"Hasil Prediksi : Memiliki penyakit Jantung")

        st.write(f"akurasi : {score}")
