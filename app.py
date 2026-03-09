import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Налаштування сторінки 
st.set_page_config(page_title="Face Detection App", layout="wide")


st.sidebar.title("⚙️ Налаштування алгоритму")
st.sidebar.markdown("**Документація (Як користуватися):**\n"
                    "1. Завантажте фотографію (JPG/PNG).\n"
                    "2. Налаштуйте параметри повзунками нижче.\n"
                    "3. Алгоритм автоматично знайде обличчя та виділить їх рамками.")

st.sidebar.subheader("Параметри Haar Cascade:")
# scaleFactor - компенсує розмір обличчя, якщо воно далеко чи близько
scale_factor = st.sidebar.slider("Масштаб (scaleFactor)", min_value=1.01, max_value=1.5, value=1.1, step=0.01,
                                 help="Визначає, наскільки зменшується розмір зображення на кожному масштабі. Менше значення = ретельніший пошук, але повільніше.")
# minNeighbors - відсіює хибні спрацювання
min_neighbors = st.sidebar.slider("Чутливість (minNeighbors)", min_value=1, max_value=15, value=5, step=1,
                                  help="Скільки сусідніх прямокутників потрібно знайти, щоб підтвердити обличчя. Більше значення = менше хибних спрацювань.")

color_choice = st.sidebar.color_picker("Колір рамки", "#00FF00")


st.title("🧑‍💻 Інтелектуальна система розпізнавання облич")
st.markdown("---")

uploaded_file = st.file_uploader("Завантажте зображення з обличчями...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    img_array = np.array(image)

    
    img_cv2 = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)

    # Завантаження вбудованого в OpenCV класифікатора Хаара
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Розпізнавання облич
    faces = face_cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors, minSize=(30, 30))

    # Конвертація HEX кольору з палітри у формат RGB для OpenCV
    h = color_choice.lstrip('#')
    rgb_color = tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


    for (x, y, w, h) in faces:
        cv2.rectangle(img_array, (x, y), (x+w, y+h), rgb_color, 3)
        cv2.putText(img_array, "Face", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, rgb_color, 2)


    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Оригінал")
        st.image(image, use_column_width=True)
    with col2:
        st.subheader(f"Результат (Знайдено облич: {len(faces)})")
        st.image(img_array, use_column_width=True)
else:
    st.info("Будь ласка, завантажте зображення, щоб розпочати роботу алгоритму.")


st.markdown("---")
st.markdown("© 2026 Розробив: **[Падус Віталій / vit20087]** | Системи штучного інтелекту")
