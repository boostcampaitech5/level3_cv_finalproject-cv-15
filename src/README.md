# 제목 없음

![image](https://github.com/boostcampaitech5/level3_cv_finalproject-cv-15/assets/59987079/0f275485-7d23-4a7f-81d7-e0d419a8acc1)

## 🐕 Dataset - 피부

---

- 출처 : AI-Hub
- 질병 클래스 종류
    - 강아지 : 6개 (’구진_플라크’, ‘비듬_각질_상피성잔고리’, ‘태선화_과다색소침착’, ‘농포_여드름’, ‘미란_궤양’, ‘결절_종괴’
    - 고양이 : 3개 (’비듬_각질_상피성잔고리’, ‘농포_여드름’, ‘결절_종괴’)
- 이미지 크기 : (1920, 1080)
- annotation : segmentation, detection이 가능하도록 구성되어 있음(points, bbox)

### 🐈 Dataset - 안구

---

- 출처 : AI-Hub
- 질병 클래스 종류
    - 강아지 : 5개 (’유루증’, ‘안검내반증’, ‘안검염’, ‘결막염’, ‘안검종양’)
    - 고양이 : 5개 (’각막궤양’, ‘각막부골편’, ‘결막염’, ‘비궤양성각막염’, ‘안검염’)
- 이미지 크기 : (400, 400)

### 🐩 Dataset - 생성 모델

---

- 출처: 인터넷 크롤링
- 국내 연예인 사진 약 1000장

### 📊 EDA 결과 - 피부

---

- class 마다 labeling(bbox, points) 크기가 상이
- 다른 class에 해당하는 질환 데이터가 여러 class에 존재
- 동일한 이미지 데이터가 여러 class에 걸쳐 다수 존재
- 라벨러의 라벨링 convention이 대체로 일치하지 않음
    
    

### 📊 EDA 결과 - 안구

---

- 다른 class에 해당하는 질환 data가 여러 class에 존재
- bbox 정보가 존재하지만, 질병에 대한 bbox 정보가 아닌 원시 데이터에서 안구를 detection할 때 사용한 bbox 정보인 것으로 확인

## 🍀 Folder Structure

---

```
├── config : 학습할때마다 바꿔가며 써야하는 부분(ex) loss, optimizer 등을 정의하는 폴더
├── data : 학습을 위한 torch의 dataset과 datamodule들 정의하는 폴더
├── loss: 학습에 사용한 loss 모음
├── models 
├── module : classification과 segmentation 학습을 위한 lightning module
├── utils : seed 설정
├── __init__.py
└── train.py

```