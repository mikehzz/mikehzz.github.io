---
layout: single
title:  "[Object Detection -6]Faster RCNN 모델"

categories:
  - Computer Vision
tags:
---


6.1 Faster RCNN 배경
---

RPN(Region Proposal Network)과 Fast RCNN의 구조를 합친것인데,  
RPN은 region proposal을 생성하는 network이다. 
R-CNN과 Fast R-CNN에서 selective search로 region proposals를 생성하고,  
각 proposals의 object class와  bounding box를 예측하기 위해 ImageNet를 사용했다.

따라서 R-CNN과 Fast R-CNN에서 region proposal 방법과 detection 신경망은 서로 분리되어 있었다.

이 둘을 분리하는 것은 좋지 않다. Selective Search가 잘못 예측을 했을 때, 이 오류는 고스란히 detection 신경망에  
직접적으로 안좋은 영향을 끼치기 때문에 이 둘을 하나로 묶는 RPN방식을 Faster RCNN에서 사용한다.

![1](/assets/images/cv-6/1.JPG)


6.2 Faster RCNN 구조
---

![2](/assets/images/cv-6/2.JPG)

앞선 모델과 달리 원본이미지에서 Selective Search를 하는 것이 아닌 feature map에서 RPN을 통해 bbox를 예측하기 때문에  
Neural Network 구조로 변경이되고, GPU 사용으로 빠른 학습과 End to End Network 학습으로 성능이 뛰어나다.

6.3 RPN
---

selective search를 대신해 영역을 추정하는 layer이다.  
입력으로 ImageNet을 통해 만들어진 featuremap을 받고 출력으로는 2가지, Object인지 아닌지 분류와 영역 Box 좌표를 내보낸다.

그러면 영역을 특정하는 방법은 무엇일까?

RPN에서 분류, 영역 좌표를 계산하기 위해서는 영역의 후보를 정하는 과정이 필요하다.

이때 사용되는 개념이 **Anchor Box**이다.

![3](/assets/images/cv-6/3.JPG)

한 지점에서 9가지 종류의 가상의 박스를 만들어 영역 후보를 만드는 과정이다.  
슬라이딩 윈도우 방식처럼 Anchor Box를 이미지 위에서 움직여가며 후보 영역을 생성한다. 이때 모든 점에서 수행하는  
것이 아니라 40x50처럼 이미지를 나누어 중심을 설정하고 설정된 중심에서 Anchor Box를 적용한다.

![4](/assets/images/cv-6/4.JPG)

다음 그림처럼 약 2000개의 지점에서 Anchor Box를 수행한다.(2000 x 9)

Anchor Box의 장점은 정사각형, 세로로 긴, 가로로 긴 물체들을 각각 탐지할수 있다는 점이다.

### 6.3.1 RPN의 구조

![5](/assets/images/cv-6/5.JPG)

RPN이 동작하는 알고리즘은 다음과 같다.

1. ImageNet을 통해 뽑아낸 feature map을 입력으로 받는다. 이때, 크기를 H x W x C로 잡는다.(각각 가로,세로,채널 수)
2. feature map에 3x3 cnn을 256채널만큼 수행한다. (위 그림에서 intermediate layer에 해당) 이때, padding을 same으로 지정해
H x W가 보존될 수 있게 해준다.(Anchor box의 좌표를 얻기 위함) 
3. 2번과정에서 출력된 feature map을 입력 받아 분류와 bbox의 예측값을 계산해야한다. 이때 1x1 conv을 이용해 계산하는 Fully Convolution
Network의 특징을 갖는다. 이는 입력 이미지의 크기에 상관없이 동작할 수 있도록 하기 위함이다.
4. 먼저 Classification을 수행하기 위해서 1 x 1 컨볼루션을 (2(오브젝트 인지 아닌지 나타내는 지표 수) x 9(앵커 개수)) 체널 수 만큼 수행해주며, 
그 결과로 H x W x 18 크기의 피쳐맵을 얻는다. H x W 상의 하나의 인덱스는 피쳐맵 상의 좌표를 의미하고,
그 아래 18개의 체널은 각각 해당 좌표를 앵커로 삼아 k개의 앵커 박스들이 object인지 아닌지에 대한 예측 값을 담고 있다.
즉, 한번의 1x1 컨볼루션으로 H x W 개의 앵커 좌표들에 대한 예측을 모두 수행한 것이다. 이제 이 값들을 적절히 reshape 해준 다음 Softmax를 적용하여
해당 앵커가 오브젝트일 확률 값을 얻는다.
5. 두 번째로 Bounding Box Regression 예측 값을 얻기 위한 1 x 1 컨볼루션을 (4 x 9) 체널 수 만큼 수행한다. 리그레션이기 때문에 결과로 얻은 값을 그대로 사용
6. 이제 앞서 얻은 값들로 RoI를 계산한다. 먼저 Classification을 통해서 얻은 물체일 확률 값들을 정렬한 다음, 높은 순으로 K개의 앵커만 추려낸다.
그 다음 K개의 앵커들에 각각 Bounding box regression을 적용해준다. 그 다음 Non-Maximum-Suppression을 적용하여 RoI을 구해준다. 















<img src="https://latex.codecogs.com/svg.latex?\Large&space;t_i^u" title="t_i^u" />는 u 클래스에 해당하는 bounding box regression offset 이다.  
<img src="https://latex.codecogs.com/svg.latex?\Large&space;v_i" title="t_i" />는 u 클래스에 해당하는 true bounding box 이다.




### 출처 : 인프런 "딥러닝 컴퓨터 비전"  
