import io
import json
import os
import time
from uuid import uuid4

import cv2
import numpy as np
import requests
import streamlit as st
from dotenv import load_dotenv
from PIL import Image
from streamlit_cropper import st_cropper
from synology_api import filestation

load_dotenv()

fl = filestation.FileStation(
    ip_address=os.environ["NAS_ADDR"],
    port=os.environ["NAS_PORT"],
    username=os.environ["NAS_ID"],
    password=os.environ["NAS_PASS"],
    secure=False,
    cert_verify=False,
    dsm_version=7,
)
st.set_page_config(
    page_title="Hype-Squad",
    page_icon="🐶",
    # layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://www.extremelycoolapp.com/help",
        "Report a bug": "https://www.extremelycoolapp.com/bug",
        "About": "# This is a header. This is an *extremely* cool app!",
    },
)

# style
st.markdown(
    """<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Serif+KR:wght@300&family=Song+Myung&display=swap');
div.stButton > button {
    color:#b4a7d6;
    font-family: 'Noto Serif KR', serif;
    padding : 13px;
    width:200px;
    margin-top: 50px;
    margin: auto;
    display:block;
    }
.title{
    font-family: 'Noto Serif KR', serif;
    text-align: center;
    font-size: 75px;
    color: #8e7cc3;
    margin-top: 10px;
    margin-bottom: 50px;
    margin-right: 35px;
}
.sub_title{
    font-family: 'Noto Serif KR', serif;
    text-align: center;
    font-size: 25px;
    color: #5b5b5b;
    margin-top: 20px;
    margin-bottom: 10px;
}
.service_process{
    font-family: 'Noto Serif KR', serif;
    text-align: center;
    font-size: 25px;
    color: #5b5b5b;
    margin-bottom: 40px;
}
.service_process2{
    font-family: 'Noto Serif KR', serif;
    text-align: center;
    font-size: 25px;
    color: #5b5b5b;
    margin-top:50px;
    margin-bottom: 40px;
}
.disease{
    font-family: 'Noto Serif KR', serif;
    text-align: left;
    font-size: 30px;
    color: #444444;
    margin-top:50px;
    margin-bottom: 20px;
}
.explain_disease{
    font-family: 'Noto Serif KR', serif;
    text-align: left;
    font-size: 18px;
    color: #444444;
    margin-top:10px;
    margin-bottom: 40px;
}
.progress-bar {
    width: 100%;
    height: 30px;
    background-color: #dedede;
    font-weight: 600;
    font-size: .8rem;
    margin-bottom: 10px;
}
.progress-bar .progress {
    width: 72%;
    height: 30px;
    padding: 0;
    text-align: center;
    background-color: #4F98FF;
    color: #111;
}
    </style>
    """,
    unsafe_allow_html=True,
)

dog_classes = {0: "유루증", 1: "안검내반증", 2: "안검염", 3: "결막염", 4: "안검종양"}
cat_classes = {0: "결막염", 1: "안검염", 2: "각막궤양", 3: "각막부골편", 4: "비궤양성각막염"}


def main():
    top1, top2 = st.columns([1, 3.5])
    logo = Image.open("./photo/hypetcare.png")
    logo = logo.resize((150, 150))
    top1.image(logo)

    top2.markdown(
        '<p class="title">HypetCare</p>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<p class="sub_title">강아지/고양이의 안구질환을 진단하는 서비스입니다.</p>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<p class="service_process">예시 사진처럼 한쪽 눈을 가까이에서 찍어주세요.</p>',
        unsafe_allow_html=True,
    )

    ex1, ex2 = st.columns(2)
    dog_eye_ex = Image.open("./photo/dog_eye_ex.jpg")
    cat_eye_ex = Image.open("./photo/cat_eye_ex.jpg")
    ex1.image(dog_eye_ex)
    ex2.image(cat_eye_ex)
    st.markdown(
        '<p class="service_process2">1. 사진을 업로드 해주세요.</p>',
        unsafe_allow_html=True,
    )
    uploaded_file = st.file_uploader(label=" ", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file)

        st.markdown(
            '<p class="service_process2">2. 눈부분만 crop해주세요.</p>',
            unsafe_allow_html=True,
        )

        c1, c2 = st.columns([1.5, 1])

        with c1:
            cropped_img = st_cropper(
                image,
                realtime_update=True,
                box_color="#00FF00",
                aspect_ratio=None,
            )

        c2.image(cropped_img, "cropped image")

        st.markdown(
            '<p class="service_process2">3. 반려동물 종을 선택해주세요.</p>',
            unsafe_allow_html=True,
        )

        dog, cat = st.columns(2)
        dog_button = dog.button("강아지")
        cat_button = cat.button("고양이")

        if dog_button:
            image_name = f"{str(uuid4())}"
            imgByteArr = io.BytesIO()
            imgByteArr = imgByteArr.getvalue()

            file_path = "./temp/" + image_name + ".jpg"

            cv2.imwrite(
                file_path,
                cv2.cvtColor(np.array(cropped_img), cv2.COLOR_BGR2RGB),
            )

            fl.upload_file(dest_path="/BoostCamp/Inference_Image", file_path=file_path)

            os.remove(file_path)

            response = requests.get(
                "http://localhost:8000/dog_eye_predict?path={image_name}",
                params={"path": image_name},
            )

            result = response.json()

            fl.get_file(
                path="/BoostCamp/Result_Image/" + image_name + "_gradcam.jpg",
                mode="download",
                dest_path="./temp",
            )

            img = cv2.imread("./temp/" + image_name + "_gradcam.jpg")
            col1, col2 = st.columns(2)

            with col1:
                st.image(img)
            with col2:
                for idx, x in enumerate(result["output"][0]):
                    value = np.linspace(0, x, num=50)
                    progress = st.progress(0, text=f"{dog_classes[idx]} : 0%")
                    if x >= 0.8:
                        st.markdown(
                            f"<style>\n"
                            f"#root > div:nth-child(1) > div.withScreencast > div > div > div > section > div.block-container.css-1y4p8pa.e1g8pov64 > div:nth-child(1) > div > div:nth-child(12) > div:nth-child(2) > div:nth-child(1) > div > div:nth-child({idx*2+1}) > div > div.st-b8 > div > div > div\n"
                            "{background-color:tomato;}\n"
                            "</style>",
                            unsafe_allow_html=True,
                        )
                    elif x >= 0.5:
                        st.markdown(
                            f"<style>\n"
                            f"#root > div:nth-child(1) > div.withScreencast > div > div > div > section > div.block-container.css-1y4p8pa.e1g8pov64 > div:nth-child(1) > div > div:nth-child(12) > div:nth-child(2) > div:nth-child(1) > div > div:nth-child({idx*2+1}) > div > div.st-b8 > div > div > div\n"
                            "{background-color:gold;}\n"
                            "</style>",
                            unsafe_allow_html=True,
                        )
                    elif x < 0.5:
                        st.markdown(
                            f"<style>\n"
                            f"#root > div:nth-child(1) > div.withScreencast > div > div > div > section > div.block-container.css-1y4p8pa.e1g8pov64 > div:nth-child(1) > div > div:nth-child(12) > div:nth-child(2) > div:nth-child(1) > div > div:nth-child({idx*2+1}) > div > div.st-b8 > div > div > div\n"
                            "{background-color:royalblue;}\n"
                            "</style>",
                            unsafe_allow_html=True,
                        )
                    for v in value:
                        time.sleep(0.01)
                        progress.progress(v, text=f"{dog_classes[idx]} : {int(v*100)}%")

            os.remove("./temp/" + image_name + "_gradcam.jpg")

            with open("disease.json", "r") as f:
                json_data = json.load(f)

            time.sleep(1)
            for idx, x in enumerate(result["output"][0]):
                if x >= 0.8:
                    st.markdown(
                        f'<p class="disease">🐶 {dog_classes[idx]} 질환이 의심됩니다.</p>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f'<p class="explain_disease">{json_data[0]["dog_eye"][dog_classes[idx]]}</p>',
                        unsafe_allow_html=True,
                    )

        elif cat_button:
            image_name = f"{str(uuid4())}"
            imgByteArr = io.BytesIO()
            imgByteArr = imgByteArr.getvalue()

            file_path = "./temp/" + image_name + ".jpg"

            cv2.imwrite(
                file_path,
                cv2.cvtColor(np.array(cropped_img), cv2.COLOR_BGR2RGB),
            )

            fl.upload_file(dest_path="/BoostCamp/Inference_Image", file_path=file_path)

            os.remove(file_path)

            response = requests.get(
                "http://localhost:8000/dog_eye_predict?path={image_name}",
                params={"path": image_name},
            )

            result = response.json()

            fl.get_file(
                path="/BoostCamp/Result_Image/" + image_name + "_gradcam.jpg",
                mode="download",
                dest_path="./temp",
            )

            img = cv2.imread("./temp/" + image_name + "_gradcam.jpg")
            col1, col2 = st.columns(2)

            with col1:
                st.image(img)
            with col2:
                for idx, x in enumerate(result["output"][0]):
                    value = np.linspace(0, x, num=50)
                    progress = st.progress(0, text=f"{cat_classes[idx]} : 0%")
                    if x >= 0.8:
                        st.markdown(
                            f"<style>\n"
                            f"#root > div:nth-child(1) > div.withScreencast > div > div > div > section > div.block-container.css-1y4p8pa.e1g8pov64 > div:nth-child(1) > div > div:nth-child(12) > div:nth-child(2) > div:nth-child(1) > div > div:nth-child({idx*2+1}) > div > div.st-b8 > div > div > div\n"
                            "{background-color:tomato;}\n"
                            "</style>",
                            unsafe_allow_html=True,
                        )
                    elif x >= 0.5:
                        st.markdown(
                            f"<style>\n"
                            f"#root > div:nth-child(1) > div.withScreencast > div > div > div > section > div.block-container.css-1y4p8pa.e1g8pov64 > div:nth-child(1) > div > div:nth-child(12) > div:nth-child(2) > div:nth-child(1) > div > div:nth-child({idx*2+1}) > div > div.st-b8 > div > div > div\n"
                            "{background-color:gold;}\n"
                            "</style>",
                            unsafe_allow_html=True,
                        )
                    elif x < 0.5:
                        st.markdown(
                            f"<style>\n"
                            f"#root > div:nth-child(1) > div.withScreencast > div > div > div > section > div.block-container.css-1y4p8pa.e1g8pov64 > div:nth-child(1) > div > div:nth-child(12) > div:nth-child(2) > div:nth-child(1) > div > div:nth-child({idx*2+1}) > div > div.st-b8 > div > div > div\n"
                            "{background-color:royalblue;}\n"
                            "</style>",
                            unsafe_allow_html=True,
                        )
                    for v in value:
                        time.sleep(0.01)
                        progress.progress(v, text=f"{cat_classes[idx]} : {int(v*100)}%")

            os.remove("./temp/" + image_name + "_gradcam.jpg")


if __name__ == "__main__":
    main()
