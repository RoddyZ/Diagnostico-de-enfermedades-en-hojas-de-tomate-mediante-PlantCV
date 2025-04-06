from typing import Optional

import requests
import streamlit as st
from app.settings import API_BASE_URL
from PIL import Image


def login(username: str, password: str) -> Optional[str]:
    """This function calls the login endpoint of the API to authenticate the user
    and get a token.

    Args:
        username (str): email of the user
        password (str): password of the user

    Returns:
        Optional[str]: token if login is successful, None otherwise
    """

    # Steps to Build the `login` Function:
    #  1. Construct the API endpoint URL using `API_BASE_URL` and `/login`.
    #  2. Set up the request headers with `accept: application/json` and
    #     `Content-Type: application/x-www-form-urlencoded`.
    #  3. Prepare the data payload with fields: `grant_type`, `username`, `password`,
    #     `scope`, `client_id`, and `client_secret`.
    #  4. Use `requests.post()` to send the API request with the URL, headers,
    #     and data payload.
    #  5. Check if the response status code is `200`.
    #  6. If successful, extract the token from the JSON response.
    #  7. Return the token if login is successful, otherwise return `None`.
    #  8. Test the function with various inputs.

    # 1: Construct the API endpoint URL
    url = f"{API_BASE_URL}/login"

    # 2: Set up the request headers
    headers = {
        "accept": "application/json",
        "Content-Type": "application/x-www-form-urlencoded"
    }

    # 3: Prepare the data payload
    data = {
        "grant_type": "",
        "username": username,
        "password": password,
        "scope": "",
        "client_id": "",
        "client_secret": ""
    }

    try:
        # 4: Use `requests.post()` to send the API request
        response = requests.post(url, headers=headers, data=data)

        # 5: Check if the response status code is 200
        if response.status_code == 200:
            # 6: Extract the token from the JSON response
            token = response.json().get("access_token")

            # 7: Return the token if login is successful
            return token

    except requests.RequestException as e:
        st.error(f"Se produjo un error al iniciar sesión: {e}")

    # 7: Return None if login fails
    return None


def predict(token: str, uploaded_file: Image) -> requests.Response:
    """This function calls the predict endpoint of the API to classify the uploaded
    image.

    Args:
        token (str): token to authenticate the user
        uploaded_file (Image): image to classify

    Returns:
        requests.Response: response from the API
    """
    # Steps to Build the `predict` Function:
    #  1. Create a dictionary with the file data. The file should be a
    #     tuple with the file name and the file content.
    #  2. Add the token to the headers.
    #  3. Make a POST request to the predict endpoint.
    #  4. Return the response.
    response = None

    # 1: Convert the image to bytes
    image_bytes = uploaded_file.getvalue()

    # 2: Create a dictionary with the file data without MIME type
    files = {
        "file": (uploaded_file.name, image_bytes)
    }

    # 3: Add only the Authorization token to headers
    headers = {
        "Authorization": f"Bearer {token}"
    }

    # 4: Construct the correct API endpoint URL
    url = f"{API_BASE_URL}/model/predict"

    try:
        # 5: Make a POST request to the predict endpoint with the file and headers
        response = requests.post(url, headers=headers, files=files)

        # 6: Return the response from the API
        return response

    except requests.RequestException as e:
        st.error(f"Ocurrió un error al hacer la predicción: {e}")
        return None



# Interfaz de usuario
st.set_page_config(page_title="Predictor Enfermedades y Especies", page_icon="📷")
st.markdown(
    "<h1 style='text-align: center; color: #4B89DC;'>Predictor Enfermedades y Especies</h1>",
    unsafe_allow_html=True,
)

# Formulario de login
if "token" not in st.session_state:
    st.markdown("## Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        token = login(username, password)
        if token:
            st.session_state.token = token
            st.success("Inicio de sesión exitoso!")
        else:
            st.error("Error al iniciar sesión. Por favor, revise sus credenciales.")
else:
    st.success("¡Has iniciado sesión!")


if "token" in st.session_state:
    token = st.session_state.token

    # Cargar imagen
    uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])

    print(type(uploaded_file))

    # Mostrar imagen escalada si se ha cargado
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Imagen subida", width=600)

    if "classification_done" not in st.session_state:
        st.session_state.classification_done = False

    # Botón de clasificación
    if st.button("Clasificar"):
        if uploaded_file is not None:
            response = predict(token, uploaded_file)
            if response.status_code == 200:
                result = response.json()
                predicted_class_enfermedad = result['predicted_class_enfermedad']
                predicted_prob_enfermedad  = result['predicted_prob_enfermedad'] * 100  # Convertir a porcentaje
                predicted_class_especie    = result['predicted_class_especie']
                predicted_prob_especie     = result['predicted_prob_especie'] * 100  # Convertir a porcentaje

                if "sana" in predicted_class_enfermedad.lower() or "sano" in predicted_class_enfermedad.lower() or predicted_class_enfermedad == "Healthy" or predicted_class_enfermedad == "healthy":
                    st.write(f"La planta de **{predicted_class_especie}** está **sana**")
                else:
                    st.write(f"La planta de **{predicted_class_especie}** está enferma de **{predicted_class_enfermedad}**")

                st.write(f"Con un nivel de certeza de : ")
                st.write(f"Enfermedad: **{predicted_prob_enfermedad:.2f}%**;   Especie: **{predicted_prob_especie:.2f}%**")
                st.session_state.classification_done = True
                st.session_state.result = result
            else:
                st.error("Error al clasificar la imagen. Por favor inténtalo de nuevo.")
        else:
            st.warning("Por favor, sube una imagen antes de clasificar.")

    # Pie de página
    st.markdown("<hr style='border:2px solid #4B89DC;'>", unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align: center; color: #4B89DC;'>2025 Predictor Enfermedades y Especies App</p>",
        unsafe_allow_html=True,
    )
