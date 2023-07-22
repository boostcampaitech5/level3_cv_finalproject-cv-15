import io
import json
import os
from uuid import uuid4

import cv2
import numpy as np
import requests
import streamlit as st
from dotenv import load_dotenv
from PIL import Image
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
    </style>""",
    unsafe_allow_html=True,
)

dog_skin_class = {
    0: "구진(플라크)",
    1: "비듬각질상피성잔고리",
    2: "태선화과다색소침착",
    3: "농포여드름",
    4: "미란궤양",
    5: "결절종괴",
}
cat_skin_class = {
    0: "상피성잔고리",
    1: "농포(여드름)",
    2: "결절종괴",
}


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
        '<p class="sub_title">강아지/고양이의 피부질환을 진단하는 서비스입니다.</p>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<p class="service_process">예시 사진처럼 증상이 의심되는 부위를 가까이에서 찍어주세요.</p>',
        unsafe_allow_html=True,
    )

    cat_ex = Image.open("./photo/cat_skin_ex.jpeg")
    st.image(cat_ex)

    st.markdown(
        '<p class="service_process2">사진을 업로드 해주세요.</p>',
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        image_name = f"{str(uuid4())}"
        imgByteArr = io.BytesIO()
        imgByteArr = imgByteArr.getvalue()

        file_path = "./temp/" + image_name + ".jpg"

        cv2.imwrite(
            file_path,
            cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB),
        )

        fl.upload_file(dest_path="/BoostCamp/Inference_Image", file_path=file_path)

        os.remove(file_path)

        st.markdown(
            '<p class="service_process2">반려동물 종을 선택해주세요.</p>',
            unsafe_allow_html=True,
        )

        dog, cat = st.columns(2)
        dog_button = dog.button("강아지")
        cat_button = cat.button("고양이")

        if dog_button:
            response = requests.get(
                "http://49.50.166.238:30008/cat_skin_seg?path={image_name}",
                params={"path": image_name},
            )

            result = response.json()

            fl.get_file(
                path="/BoostCamp/Result_Image/" + image_name + "_seg.jpg",
                mode="download",
                dest_path="./temp",
            )

            img = cv2.imread("./temp/" + image_name + "_seg.jpg")
            st.image(img)
            st.markdown(
                f'<p class="disease">🐶 {cat_skin_class[int(result["output"])]} 질환이 의심됩니다.</p>',
                unsafe_allow_html=True,
            )
            with open("disease.json", "r") as f:
                json_data = json.load(f)

            st.markdown(
                f'<p class="explain_disease">{json_data[0]["dog_skin"][cat_skin_class[int(result["output"])]]}</p>',
                unsafe_allow_html=True,
            )

        elif cat_button:
            response = requests.get(
                "http://localhost:8000/cat_skin_seg?path={image_name}",
                params={"path": image_name},
            )

            result = response.json()

            fl.get_file(
                path="/BoostCamp/Result_Image/" + image_name + "_seg.jpg",
                mode="download",
                dest_path="./temp",
            )

            img = cv2.imread("./temp/" + image_name + "_seg.jpg")
            st.image(img)
            st.markdown(
                f'<p class="disease">😺 {cat_skin_class[int(result["output"])]} 질환이 의심됩니다.</p>',
                unsafe_allow_html=True,
            )
            with open("disease.json", "r") as f:
                json_data = json.load(f)

            st.markdown(
                f'<p class="explain_disease">{json_data[0]["cat_skin"][cat_skin_class[int(result["output"])]]}</p>',
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    main()
