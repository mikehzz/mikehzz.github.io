---
layout: single
title:  "심층 신경망(Deep Neual Network)"  

categories:
  - 딥러닝 이론
tags:
  - 신경망 추론
  - 손실함수
---


**이 글은 밑바닥부터 시작하는 딥러닝 2 책을 토대로 작성**

[밑바닥부터 시작하는 딥러닝 2](https://github.com/WegraLee/deep-learning-from-scratch-2)

***********
1.1 신경망의 추론
---

### 1.1.1 신경망 추론 전체 그림

![1](/assets/images/DNN/1.PNG)

화살표에는 **가중치**가 존재하며, 그 가중치와 뉴런의 값을 각각 곱해서 그 합이 다음 뉴런의 입력으로 쓰인다.

위 그림의 신경망은 인접하는 층의 모든 뉴런과 연결되어 있다는 점에서 **완전연결계층**이라고 한다.

신경망이 수행하는 계산 수식

![2](/assets/images/DNN/2.PNG)

```python
import numpy as np
W1 = np.random.randn(2,4)
b1 = np.random.randn(4)
x = np.random.randn(10m2)
h = np.matmul(x,W1) + b1
```

여기서 완전연결계층에 의한 변환은 **선형** 변환이다.

여기에 **비선형** 효과를 부여하는 것이 바로 활성화 함수이다.

![3](/assets/images/DNN/3.PNG)


```python
import numpy as np

def sigmoid(x):
return 1/1+np.exp(-x))

x = np.random.randn(10,2)
W1 = np.random.randn(2,4)
b1 = np.random.randn(4)
W2 = np.random.randn(4,3)
b2 = np.random.randn(3)

h = np.matmul(x,W1) + b1
a = sigmoid(h)
s = np.matmul(a,W2) + b2
```

x의 형태 (10,2) - 2차원 데이터 10개를 미니 배치(다수의 샘플 데이터를 한 번에 처리)로 처리하여

s의 형태 (10,3) - 각 데이터가 3차원 데이터로 변환(각 클래스에 대응하는 점수로 분류 가능)

1.2 신경망의 학습
---
최적의 매개변수(가중치)를 찾는 작업


### 1.2.1 손실 함수(Loss function)

1. 신경망의 성능을 나타내는 척도

2. 학습 데이터(정답 데이터)와 신경망이 예측한 결과를 비교하여 예측이 얼마나 나쁜가 산출한 단일 값(스칼라)

교차 엔트로피 오차(Cross Entropy Error)

![4](/assets/images/DNN/4.PNG)

softmax - 확률 출력

![5](/assets/images/DNN/5.PNG)

여기서 s는 score값 (score값이 높은 레이블이 target으로 예측할 확률이 높다)

Croos Entropy Error - 확률, 정답 레이블 입력

![6](/assets/images/DNN/6.PNG)

t_k - k번째 클래스에 해당하는 정답 레이블

정답 레이블은 t = [0,0,1] 과 같이 one-hot 벡터로 표기

즉, 어떤 데이터를 예측한 결과 값과 실제 결과 값을 비교 및 계산하여 LOSS를 구하고

LOSS를 줄이면서 학습을 한다. 


### 1.2.2 미분과 기울기

앞 절에서 LOSS를 줄이면서 학습을 한다고 했다.

LOSS를 줄이는 방법은 LOSS의 변화량(기울기)를 구하여

가중치를 갱신해 나가는 것이다.

Weight의 변화량에 따른 LOSS의 변화량을 계산한 다음

가중치를 갱신한다. (기울기가 낮은 쪽으로)


### 1.2.3 가중치 갱신

신경망 학습의 순서

1. 미니 배치      - 훈련 데이터 중 무작위로 데이터 선택 
2. 기울기 계산    - back propagation으로 가중치 매개변수에 대한 loss function의 gradient 구하기
3. 매개변수 갱신  - gradient를 사용하여 가중치 매개변수 갱신
4. 반복

경사 하강법 - 함수의 기울기를 구하여 낮은 쪽으로 계속 이동시켜 극값에 이를 때까지 반복하는 방법

매개변수 갱신 기법

- 확률적 경사 하강법(SGD)

무작위로 선택된 데이터에 대한 기울기를 이용한다.

![7](/assets/images/DNN/7.PNG)

W - 갱신하는 가중치 매개변수

n(에타)는 학습률(learning rate)을 나타낸다.


### 에폭(epoch), 배치 사이즈(batch size), 미니배치(mini batch), 이터레이션(iteration)

에폭(epoch): 하나의 단위. 1에폭은 학습에서 훈련 데이터를 모두 소진했을 때의 횟수에 해당한다.

미니 배치(mini batch): 훈련 데이터 셋을 몇 개의 데이터 셋으로 나누었을 때, 그 작은 데이터 셋 뭉치

배치 사이즈(batch size): 하나의 미니 배치에 넘겨주는 데이터 갯수, 즉 한번의 배치마다 주는 샘플의 size

이터레이션(iteration): 하나의 미니 배치를 학습할 때 1iteration이라고 한다. 즉, 미니 배치 갯수 = 이터레이션 수

![8](/assets/images/DNN/8.PNG)

[출처](https://mole-starseeker.tistory.com/59)

예) 훈련 데이터 1000개를 10개의 미니 배치로 나누어 학습하는 경우, 배치 사이즈는 100이 되고, 
확률적 경사 하강법(SGD)을 10회 반복, 즉 10회 iteration 하면 모든 훈련 데이터를 '소진'하게 된다. 
이때 SGD 10회, 즉 iteration 10회가 1에폭이 된다. 
그러면 10에폭을 돌린다고 했을때 가중치 갱신은 10x10 = 100회가 일어난다.

1.3 정리
---

- 신경망에는 입력층, 은닉층, 출력층이 존재한다.
- 각 연결층에 의한 변환은 비선형 으로 부여하고 활성화 함수로 이용한다.
- 신경망을 학습을 한다는 것은 최적의 매개변수(가중치)를 찾는 작업이다.
- 출력층에 나온 score값과 실제 값을 비교하여 손실함수를 구하고, 손실을
줄여 나가면서 최적의 매개변수를 찾는 것이다.
