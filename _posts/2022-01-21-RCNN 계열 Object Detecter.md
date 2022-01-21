---
layout: single
title:  "[Object Detection -3]RCNN 계열 Object Detecter(RCNN, SPPNet, Fast RCNN, Faster RCNN"

categories:
  - Computer Vision
tags:
  - 
---

3.1 Object Localization / Detection 개요
---

Object Localization / Detection은 이미지에서 객체의 위치를 찾고, Class를 맞추는 것이다.  
개요는 다음과 같다.

- Object Localization/Detection
  - 원본 이미지에서 특정 Object의 위치를 찾는 것이다.
  - Image Classification + Bounding Box Regression
  - Image Classification과 동일한 과정
    - 원본 이미지 >> Feature Extrator >> Feature Map >> FC Layer >> Soft max
  - Bounding Box Regression
    - Bounding Box 좌표값(위치)를 구하는 과정

3.2 RCNN
---

### 3.2.1 RCNN 개요

RCNN은 이미지의 위치를 Region Proposal 방식으로 찾게된다.  
즉, 원본 이미지에서 물체가 있을법한 위치 2000개(bbox)를 찾고, **crop과 warp을 적용해** bbox를 ImageNet을 통해 Feature Extractor로 feature를  
추출하고, Feature Map을 FC layer로 나열한 다음 SVM Classifier로 분류, bbox Regreesion을 통해 위치를 찾게된다.

![1](/assets/images/cv-3/1.JPG)

다음과 같은 과정을 거치는 것이 RCNN의 모델이다.

이 과정을 이미지로 표현하면,

![2](/assets/images/cv-3/2.JPG)

여기서 crop, warp을 하는 이유는 Classification Dense layer로 인해 이미지 크기가 동일 해야하기 때문이다.

### 3.2.2 RCNN 학습

RCNN 학습과정은 다음과 같다.

![3](/assets/images/cv-3/3.JPG)

원본이미지에 Selective Search를 적용해 얻은 bbox를 GT(Ground Truth)와 IOU를 구해 0.5 이상인 경우에만  
해당 클래스로 적용하고, 나머지는 Background로 fine-tuning한다.  

### 3.2.3 RCNN 문제점

- 네트워크를 학습시키는데 방대한 시간 소요
  - 하나의 이미지에 대해 2000개의 Region Proposal을 분류
  - 하나의 테스트 이미지에 소요되는 시간 약 47초
  - 실제 사례에 적용할 수 없을 정도로 느린 시간
- 추론 시간도 느림
  - 2000개 각각에 대한 추론 진행

### 출처 : 인프런 "딥러닝 컴퓨터 비전"  
