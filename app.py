#Kütüphaneler
import streamlit as st 
import cv2
from PIL import Image,ImageEnhance
import numpy as np 
import os


face_cascade = cv2.CascadeClassifier('model/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('model/haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('model/haarcascade_smile.xml')


def detect_faces(resim_dosya):
  new_img = np.array(resim_dosya.convert('RGB'))
  img = cv2.cvtColor(new_img,1)
  gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
  # Detect faces
  faces = face_cascade.detectMultiScale(gray, 1.1, 4)
  # Draw rectangle around the faces
  for (x, y, w, h) in faces:
         cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
  return img,faces


def detect_eyes(our_image):
  new_img = np.array(our_image.convert('RGB'))
  img = cv2.cvtColor(new_img,1)
  gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
  eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)
  for (ex,ey,ew,eh) in eyes:
          cv2.rectangle(img,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
  return img

def detect_smiles(our_image):
  new_img = np.array(our_image.convert('RGB'))
  img = cv2.cvtColor(new_img,1)
  gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
  # Detect Smiles
  smiles = smile_cascade.detectMultiScale(gray, 1.1, 4)
  # Draw rectangle around the Smiles
  for (x, y, w, h) in smiles:
      cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
  return img

def cartonize_image(our_image):
  new_img = np.array(our_image.convert('RGB'))
  img = cv2.cvtColor(new_img,1)
  gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
  # Edges
  gray = cv2.medianBlur(gray, 5)
  edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
  #Color
  color = cv2.bilateralFilter(img, 9, 300, 300)
  #Cartoon
  cartoon = cv2.bitwise_and(color, color, mask=edges)

  return cartoon


def cannize_image(our_image):
  new_img = np.array(our_image.convert('RGB'))
  img = cv2.cvtColor(new_img,1)
  img = cv2.GaussianBlur(img, (11, 11), 0)
  canny = cv2.Canny(img, 100, 150)
  return canny


def main():
	"""Face Detection Application"""

	st.title("Yuz Tanima Uygulamasi")
	st.title("Streamlit ve OpenCV ile gelistirilmistir")

	islemler = ["Tespit Et","Hakkinda"]
	secim = st.sidebar.selectbox("Lutfen Seciniz",islemler)

	if secim == 'Hakkinda':
		st.subheader("Hakkinda")
		st.write("Hamdi Damar tarafından geliştirilmiştir.")


	if secim == 'Tespit Et':
		resim_dosya = st.file_uploader("Resim Yukle",type=["jpg","png","jpeg"])

		if resim_dosya is not None:
			st.write("Yükleme Başarılı!")
			resim = Image.open(resim_dosya)
			st.text("Orjinal Fotograf")
			st.image(resim)

		gelistirme_turu = st.sidebar.radio("Gelistirme Turu",["Orjinal","Gri-Olcekli","Kontrast","Parlaklik","Bulaniklastirma"])


		if gelistirme_turu == 'Gri-Olcekli':
			yeni_resim = np.array(resim.convert('RGB'))
			img = cv2.cvtColor(yeni_resim,1)
			gri_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
			st.write("Islenen Fotograf")
			st.image(gri_img)

		elif gelistirme_turu == 'Kontrast':
			k_oran = st.sidebar.slider("Kontrast",0.5,3.5)
			arttirici = ImageEnhance.Contrast(resim)
			img_cikis = arttirici.enhance(k_oran)
			st.write("Islenen Fotograf")
			st.image(img_cikis)

		elif gelistirme_turu == 'Parlaklik':
			k_oran = st.sidebar.slider("Parlaklik",0.5,3.5)
			arttirici = ImageEnhance.Brightness(resim)
			img_cikis = arttirici.enhance(k_oran)
			st.write("Islenen Fotograf")
			st.image(img_cikis)

		elif gelistirme_turu == 'Bulaniklastirma':
			yeni_resim = np.array(resim.convert('RGB'))
			B_oran = st.sidebar.slider("Bulaniklik",0.5,3.5)
			img = cv2.cvtColor(yeni_resim,1)
			bulanik_img = cv2.GaussianBlur(img,(11,11),B_oran)
			st.write("Islenen Fotograf")
			st.image(bulanik_img)
		else:
			st.image(resim,width=300)

      

		gorev = ["Yuz","Gulumseme","Goz","Cannize","Karikatur"]
		ozellik_secim = st.sidebar.selectbox("Ozellik Seciniz..",gorev)

		if st.button("Algila"):

		    if ozellik_secim == 'Yuz':
		        result_img,result_faces = detect_faces(resim)
		        st.image(result_img)
		        st.success("Found {} faces".format(len(result_faces)))

		    elif ozellik_secim == 'Gulumseme':
		    	result_img = detect_smiles(resim)
		    	st.image(result_img)

		    elif ozellik_secim == 'Goz':
		    	result_img = detect_eyes(resim)
		    	st.image(result_img)

		    elif ozellik_secim == 'Karikatur':
		    	result_img = cartonize_image(resim)
		    	st.image(result_img)

		    elif ozellik_secim == 'Cannize':
		    	result_img = cannize_image(resim)
		    	st.image(result_img)

		    

if __name__ == '__main__':
	main()