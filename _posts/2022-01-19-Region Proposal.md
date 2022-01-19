---
layout: single
title:  "[Object Detection -1]Object Detection의 이해"

categories:
  - Computer Vision
tags:
  - 
---

1.1 Object Detection 이란?
---
Object Detection은 객체를 감지하는 알고리즘으로 다음과 같이 나뉠수 있다.

![1](/assets/images/cv-1/1.JPG)

- Classification : 이미지상의 객체를 분류(한개)
- Localization : 단 하나의 Object 위치를 Bounding Box로 지정하여 찾음
- Object Detection : 여러 개의 Object들에 대한 위치를 Bounding Box로 지정하여 찾음
- Segmentation : Detection보다 더 발전된 형태로 픽셀 레벨 Detection 수행

1.2 Region Proposal
---

Object Detection의 주요 구성 요소는 다음과 같다.

![2](/assets/images/cv-1/2.JPG)

여기서 Region Proposal는 영역 추정이라는 말을 쓰는데,  
주어진 이미지에서 물체가 있을 법한 위치를 찾는 것이다.  
물체의 위치를 찾으려면 일단 이미지상에서 물체가 있을 법한 Bounding box를 찾는다.

Region Proposal(영역 추정) 기법이 나오기 전에는 다음과 같은 방식을 사용했다.

![3](/assets/images/cv-1/3.JPG)

Window를 왼쪽 상단에서 부터 오른쪽 하단으로 이동시키면서 Object를 Detection하는 방식이다.

이제 Region Proposal의 방식 중 대표적인 방법인 Selective Search를 알아보자.

![4](/assets/images/cv-1/4.JPG)

원본이미지에서 후보 Bounding Box를 선택해 최종 Object Detection을 하는 과정이다.  
우리가 이미지를 직관적으로 봤을 때 객체들은 edge나 밝기의 차이 등으로 인식할 수 있다.  
이러한 점을 활용한 그래프이론 알고리즘을 통해 Bounding box 제안한다.  
즉, 컬러,무늬,크기,형태에 따라 유사도가 비슷한 Segment들을 그룹핑한다.

![5](/assets/images/cv-1/5.JPG)

다음과 같이 개별 Segment된 모든 부분들을 Bounding box로 만들고, 곂치는 box를 줄여나가면서 찾게된다.

1.3 IoU
---

IoU(Intersection over Union)는 Object Detection에서 사용되는 도구이다. 성능 지표는 아니고,  
객체 인식 모델의 성능 평가를 하는 과정에서 사용되는 도구로 생각하면 된다.

![6](/assets/images/cv-1/6.JPG)

IoU = 교집합 영역 넓이 / 합집합 영역 넓이

![7](/assets/images/cv-1/7.JPG)

1.4 mAP
---

Object Detection의 성능 평가지표인 mAP는 실제 Object가 Detected된 재현율(Recall)의 변화에  
따른 정밀도(Presion)의 값을 평균한 성능 수치이다.  
그 전에 우선 precision, recall, AP(Average Precision)에 대해 이해해야 한다.

- Precision(정밀도) : 예측을 Positive로 한 대상 중에 예측과 실제 값이 Positive로 일치한 데이터의 비율을 뜻한다.  
Object Detection에서는 검출 알고리즘이 검출 예측한 결과가 실제 Object들과 얼마나 일치하는지를 나타내는 지표이다.
- Recall(재현율) : 재현율은 실제 값이 Positive인 대상 중에 예측과 실제 값이 Positive로 일치한 데이터의 비율을 뜻한다.  
Object Detection에서는 검출 알고리즘이 실제 Object들을 빠뜨리지 않고 얼마나 정확히 검출 예측하는지를 나타내는 지표이다.

![8](/assets/images/cv-1/8.JPG)

### 오차행렬(Confusion Matrix)

오차 행렬은 이진 분류의 예측 오류가 얼마인지와 더불어 어떠한 유형의 예측 오류가 발생하고 있는지를 함께 나타내는 지표이다.

![9](/assets/images/cv-1/9.JPG)

Object Detection에서 TP는 실제 객체를 잘 예측한 경우를 말하고,  
FP는 실제 객체를 잘못된 객체로 인식하거나, IOU가 0.5미만이거나, bounding box를 잘못 선정한 경우를 뜻한다.  
FN은 실제 객체를 예측하지 못한 경우를 뜻한다.

- 정밀도 = TP / (FP + TP)
- 재현율 = TP / (FN + TP)

정밀도를 높게 만드는 법은 확실한 기준이 되는 경우에만 Positive로 예측하면 된다.(Confidence을 높게)  
재현율을 높게 만드는 법은 모든 객체를 Positive로 예측하면 된다. 즉, 난사라고 이해하면 된다.(Confidence을 낮게)

즉, Confidence 임계값에 따라 정밀도와 재현율의 값이 변화된다.

#### 예시  
Confidence에 따른 Precision과 Recall의 변화를 보면

![10](/assets/images/cv-1/10.JPG)

AP(Average Precision)을 계산해 보면

![11](/assets/images/cv-1/11.JPG)

다음 사각형의 너비가 AP이고, 물체 클래스가 여러개인 경우 각 클래스당 AP를 구한 다음에 모두 합하고 클래스의 개수로  
나눠줌으로써 성능을 평가한다. 그리고 이 평가지수를 mAP(mean Average Precision)이라고 한다.

다음 링크를 들어가면 Colab환경에서 Selective Search를 구현한 것을 확인 할 수 있다.

[깃허브 링크](https://github.com/mikehzz/Computer_Vision/blob/main/selective_search_n_iou.ipynb)

### 출처 : 인프런 "딥러닝 컴퓨터 비전"  
