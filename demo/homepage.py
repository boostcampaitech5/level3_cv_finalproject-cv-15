import os
import webbrowser

# from tkinter import PAGES
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
    initial_sidebar_state="collapsed",
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
    color: #8e7cc3;
    margin-top: 50px;
    margin-bottom: 50px;
}
div.stButton > button {
    color:#b4a7d6;
    font-family: 'Noto Serif KR', serif;
    padding : 13px;
    width:200px;
    margin-top: 50px;
    margin: auto;
    display:block;
    }
</style>""",
    unsafe_allow_html=True,
)


def main():
    top1, top2 = st.columns([1, 3.5])
    logo = Image.open("./photo/hypetcare.png")
    logo = logo.resize((150, 150))
    top1.image(logo)

    top2.markdown(
        '<p class="title">HypetCare</p>',
        unsafe_allow_html=True,
    )

    cat_img = Image.open("./photo/AI_cat.jpeg")
    dog_img = Image.open("./photo/AI_dog.jpeg")
    img1, img2 = st.columns(2)
    img1.image(cat_img)
    img2.image(dog_img)

    st.markdown(
        '<p class="sub_title">“\n원하는 서비스를 선택하세요.”</p>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    eye_service = col1.button(
        "반려동물 안구\n\n 질병 예측 서비스",
    )

    skin_service = col2.button(
        "반려동물 피부\n\n 질병 예측 서비스",
    )
    # gan_service = col3.button(
    #     "반려동물 프로필 사진\n\n 생성 서비스",
    # )

    if eye_service:
        link = "http://localhost:8502/"
        webbrowser.open_new_tab(link)

    elif skin_service:
        link = "http://localhost:8503/"
        webbrowser.open_new_tab(link)


if __name__ == "__main__":
    main()
