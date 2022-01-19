---
layout: single
title:  "[Object Detection -2]Object Detection 주요 데이터 세트 및 OpenCV 소개"

categories:
  - Computer Vision
tags:
  - 
---

2.1 Object Detection 주요 데이터 세트 소개
---

### 2.1.1 Object Detection 주요 Dataset

- PASCAL VOC
  - XML Format
  - 20개의 오브젝트 카테고리
- MS COCO
  - json Format
  - 80개의 오브젝트 카테고리: 여러가지 이미지
  - PASCAL의 평이한 오브젝트 문제 해결
- Google Open Images
  - csv Format
  - 600개의 오브젝트 카테고리

2.2 PASCAL VOC Dataset
---


PASCAL VOC 데이터셋의 구조는 다음과 같다.

![1](/assets/images/cv-2/1.JPG)

VOC 2012기준으로 

- Annotations : xml 포맷이며, 개별 xml파일은 한 개 image에 대한 Annotation 정보를 가지고 있다. 
즉, 이미지에 대한 정보라고 생각하면된다.
- ImageSet : 어떤 이미지를 train, test, trainval, val에 사용할 것인지에 대한 매핑 정보를 개별 오브젝트별로 파일로 가지고 있다.
- JPEGImages : Detection과 Segmanetation에 사용될 원본 이미지
- SegmentationClass : Semantic Segmentation에 사용될 masking 이미지
- SegmentationObject : Instance Segmentation에 사용될 masking 이미지

2.3 MS-COCO Dataset
---

### 2.3.1 MS-COCO Dataset

MS-COCO Dataset은 가장 대표적인 Dataset으로 불린다.  

- 80개의 Object Category
- 30만 개의 image들과 150만 개의 object들
- 하나의 image에 평균 5개의 object들로 구성
- Tensorflow Object Detection API 및 많은 오픈 소스 계열의 주요 패키지
- COCO Dataset으로 Pretrained된 모델 제공

![2](/assets/images/cv-2/2.JPG)

빈 ID가 있기 때문에 실제 카테고리 수는 80개이다.

### 2.3.2 MS-COCO Dataset 구성

- COCO 2017 Dataset 기준

![3](/assets/images/cv-2/3.JPG)

JSON Annotation 파일은 JSON 파일 하나, images/annotaions는 이미지 파일과 1:1 매핑 형식으로 되어 있다.

### 2.3.3 MS-COCO Dataset 특징

- 이미지 한 개에 여러 오브젝트를 가진다.
- 타 Dataset에 비해 난이도가 높은 데이터를 제공한다.
- mAP에서 보다 엄격한 기준을 가진다.

![4](/assets/images/cv-2/4.JPG)

2.4 OpenCV
---






다음 링크를 들어가면 Colab환경에서 Selective Search를 구현한 것을 확인 할 수 있다.

[깃허브 링크](https://github.com/mikehzz/Computer_Vision/blob/main/selective_search_n_iou.ipynb)

### 출처 : 인프런 "딥러닝 컴퓨터 비전"  
