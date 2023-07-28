# 제목 없음

[https://github.com/boostcampaitech5/level2_cv_semanticsegmentation-cv-15/assets/113486402/d3bca362-3b1d-4a6a-8982-6836656793f5](https://github.com/boostcampaitech5/level2_cv_semanticsegmentation-cv-15/assets/113486402/d3bca362-3b1d-4a6a-8982-6836656793f5)

## 🐾 HYPETCARE - 반려동물 질병 예측 서비스

---

- 동물 병원 방문이 쉽지 않은 일반 사용자들에게 스마트폰으로 촬영한 사진으로 간편하게 안구, 피부 질병에 대해 진단 받아 반려동물의 건강을 집에서도 예측하고 관리할 수 있다.
- 반려동물을 디즈니 공주 캐릭터로 바꿔주는 생성 모델을 사용해볼 수 있다.

## 👀 시연 영상

---

- 링크 예정

## 👨🏻‍💻 👩🏻‍💻 팀 구성

---

```python
-------------
|![logo1](https://github.com/boostcampaitech5/level2_objectdetection-cv-15/assets/99079272/53873dd9-69cc-4fe6-ba8f-034d8860cefe)|![logo2](https://github.com/suhyehye/Machine-Learning/assets/99079272/72269432-21c6-4176-90cb-cbaaa2bf0b84)|![logo3](https://github.com/suhyehye/Machine-Learning/assets/99079272/15cadbf0-98ed-4f93-86df-9a0a1aacc0ba)|![logo4](https://github.com/suhyehye/Machine-Learning/assets/99079272/fb487344-cf32-4eda-8639-f58039c7c14b)|![logo5](https://github.com/suhyehye/Machine-Learning/assets/99079272/03fa4a9d-dac4-42b5-ab5e-432e7fce8935)|
| --- | --- | --- | --- |  --- |
| [김용우](https://github.com/yongwookim1) | [박종서](https://github.com/justinpark820) | [서영덕](https://github.com/SeoYoungDeok) |[신현준](https://github.com/june95) |[조수혜](https://github.com/suhyehye) |
```

### **📆** 프로젝트 일정 : 2023.06.30 ~ 2023.07.28

---

<img width="1070" alt="Untitled" src="https://github.com/boostcampaitech5/level3_cv_finalproject-cv-15/assets/59987079/fdfaf107-97d9-47f0-ad01-dc45c656a99b">


### 📲 서비스 아키텍쳐

---

![Untitled 1](https://github.com/boostcampaitech5/level3_cv_finalproject-cv-15/assets/59987079/40fe3a6e-3dd6-4671-90fe-2d34f0f123af)


## 🍀 Folder Structure

---

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

---

- 강아지 피부 : ConvNext(encoder), Cascade-RCNN(decoder)
- 고양이 피부 : MANet(encoder), HRNet(decoder)
- 강아지 안구 : ResNest50
- 고양이 안구 : ViT_small_patch16
- 안구 detection : Yolov3(backbone: darknet53)
- 생성 모델 : DualStyleGAN

## 🔍 Reference 및 출처

---

- pytorch lightining : [https://github.com/Lightning-AI/lightning](https://github.com/Lightning-AI/lightning)
- hydra-zen : [https://github.com/mit-ll-responsible-ai/hydra-zen](https://github.com/mit-ll-responsible-ai/hydra-zen)
- mmdetection : [https://github.com/open-mmlab/mmdetection](https://github.com/open-mmlab/mmdetection)
- segmentation_models_pytorch : [https://github.com/qubvel/segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch)
- huggingface : [https://huggingface.co/](https://huggingface.co/)
- dual style gan : [https://github.com/williamyang1991/DualStyleGAN](https://github.com/williamyang1991/DualStyleGAN)
- AI hub : [https://aihub.or.kr/](https://aihub.or.kr/)