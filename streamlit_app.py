import cv2
import streamlit as st
import numpy as np

def apply_mosaic(image, faces):
    for (x, y, w, h) in faces:
        face = image[y:y+h, x:x+w]
        face = cv2.resize(face, (20, 20), interpolation=cv2.INTER_NEAREST)
        face = cv2.resize(face, (w, h), interpolation=cv2.INTER_NEAREST)
        image[y:y+h, x:x+w] = face
    return image

def main():
    st.title("Real-time Face Mosaic App📸")

    # カメラの起動
    cap = cv2.VideoCapture(0)

    # 顔検出用の分類器を読み込み
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while True:
        # カメラからフレームを取得
        ret, frame = cap.read()

        # 顔を検出
        faces = face_cascade.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), scaleFactor=1.3, minNeighbors=5)

        # 検出された顔にモザイクをかける
        frame_with_mosaic = apply_mosaic(frame.copy(), faces)

        # Streamlitウィジェットにフレームを表示
        st.image(frame_with_mosaic, caption="Mosaic Applied Image", use_column_width=True)

if __name__ == "__main__":
    main()
