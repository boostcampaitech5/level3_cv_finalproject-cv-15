![image](https://github.com/boostcampaitech5/level3_cv_finalproject-cv-15/assets/59987079/aac9514b-96eb-45b7-a1a0-e526406675e9)

## 🐾 HYPETCARE - 반려동물 질병 예측 서비스

- 동물 병원 방문이 쉽지 않은 일반 사용자들에게 스마트폰으로 촬영한 사진으로 간편하게 안구, 피부 질병에 대해 진단 받아 반려동물의 건강을 집에서도 예측하고 관리할 수 있다.
- 반려동물을 디즈니 공주 캐릭터로 바꿔주는 생성 모델을 사용해볼 수 있다.

## 👀 시연 영상

[![Video Label][http://img.youtube.com/vi/XNFxTloiSM4/0.jpg]](https://youtu.be/XNFxTloiSM4)

## 👨🏻‍💻 👩🏻‍💻 팀 구성

-------------
|![logo1](https://github.com/boostcampaitech5/level3_cv_finalproject-cv-15/assets/59987079/31b76c86-6554-49a7-ac6b-0de02b6b815b)|![logo2](https://github.com/boostcampaitech5/level3_cv_finalproject-cv-15/assets/59987079/aad157e9-746a-4bce-8387-c77d5b0018c1)|![logo3](https://github.com/boostcampaitech5/level3_cv_finalproject-cv-15/assets/59987079/27320948-6273-4caf-897a-2061dd700427)|![logo4](https://github.com/boostcampaitech5/level3_cv_finalproject-cv-15/assets/59987079/9dcf63a8-6956-42c2-b815-fc8a6117e707)|![logo5](https://github.com/boostcampaitech5/level3_cv_finalproject-cv-15/assets/59987079/640a3382-9c34-45fc-9431-c7b8926ad6fa)|
| --- | --- | --- | --- |  --- |
| [김용우](https://github.com/yongwookim1) | [박종서](https://github.com/justinpark820) | [서영덕](https://github.com/SeoYoungDeok) |[신현준](https://github.com/june95) |[조수혜](https://github.com/suhyehye) |


### **📆** 프로젝트 일정 : 2023.06.30 ~ 2023.07.28

<img width="1070" alt="Untitled (1)" src="https://github.com/boostcampaitech5/level3_cv_finalproject-cv-15/assets/59987079/92fb1fa9-2df9-44fc-8266-abd9d2bcaeac">


### 📲 서비스 아키텍쳐

![Untitled (2)](https://github.com/boostcampaitech5/level3_cv_finalproject-cv-15/assets/59987079/372a9313-ba1e-450c-88f8-a3f7fc0a426e)


## 🍀 Folder Structure

```
├── DualStyleGAN  
├── api 
│   └── app.py
├── demo : streamlit 코드와 front에 사용한 이미지
├── src : 모델 학습을 위한 코드 => lightning + hydra zen
│   ├── __init__.py
│   ├── config
│   ├── data
│   ├── loss
│   ├── model
│   ├── module
│   ├── utils
│   └── train.py
├── worker : 서버들간 연결을 위해 사용한 backend 코드
│   ├── utils : gradcam을 사용하기 위함
│   │   ├── __init__.py
│   │   └── gradcam.py
│   ├── __init__.py
│   ├── cat_skin_worker.py
│   ├── cat_eyes_worker.py
│   ├── dog_skin_worker.py
│   ├── dog_eyes_worker.py
│   └── gan_model_worker.py
├── web : 웹페이지를 구성하기 위해 사용한 frontend 코드
├── poetry.toml
├── pyproject.toml
├── run.py
└── .gitignore
```

## 💫 Final Model

- 강아지 피부 : ConvNext(encoder), Cascade-RCNN(decoder)
- 고양이 피부 : MANet(encoder), HRNet(decoder)
- 강아지 안구 : ResNest50
- 고양이 안구 : ViT_small_patch16
- 안구 detection : Yolov3(backbone: darknet53)
- 생성 모델 : DualStyleGAN

## 🔍 Reference 및 출처

- pytorch lightining : [https://github.com/Lightning-AI/lightning](https://github.com/Lightning-AI/lightning)
- hydra-zen : [https://github.com/mit-ll-responsible-ai/hydra-zen](https://github.com/mit-ll-responsible-ai/hydra-zen)
- mmdetection : [https://github.com/open-mmlab/mmdetection](https://github.com/open-mmlab/mmdetection)
- segmentation_models_pytorch : [https://github.com/qubvel/segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch)
- huggingface : [https://huggingface.co/](https://huggingface.co/)
- dual style gan : [https://github.com/williamyang1991/DualStyleGAN](https://github.com/williamyang1991/DualStyleGAN)
- AI hub : [https://aihub.or.kr/](https://aihub.or.kr/)
