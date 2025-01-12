---
layout: single
title:  "[딥러닝 1-3]신경망(Neural Network) 학습"

categories:
  - 딥러닝 이론
tags:
  - 손실함수
  - 미분
  - 경사하강법
  - 
---

**이 글은 밑바닥부터 시작하는 딥러닝 1 책을 참고로 작성**

[밑바닥부터 시작하는 딥러닝 1](https://github.com/WegraLee/deep-learning-from-scratch)


3.1 신경망 학습
---

신경망에서의 학습이란 훈련 데이터로부터 가중치 매개변수의 최적값을 자동으로 획득하는 것을 뜻한다.
### 3.1.1 미니배치

훈련 데이터 중 일부를 무작위로 가져온다.  
이렇게 선별한 데이터를 미니배치라 하며, 그 미니배치의 손실 함수 값을 줄이는 것이 목표이다.

### 3.1.2 손실함수

신경망은 '하나의 지표'를 기준으로 최적의 매개변수 값을 탐색한다. 이 지표를  **손실 함수**  
라고 한다. 손실함수는 일반적으로 **평균 제곱 오차**와 **교차 엔트로피 오차**를 사용한다.

1. 평균 제곱 오차(MSE)

수식은 다음과 같다.

![13](/assets/images/DNN_2/13.PNG)

여기서 yk는 신경망의 출력(예측값), tk는 정답 레이블, k는 데이터 차원수를 나타낸다.  
예를 들면,

```python
y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0] # 소프트맥스 함수의 출력
t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]                       # 정답을 가르키는 위치의 원소
```

2. 교차 엔트로피 오차(Cross entropy error)

![14](/assets/images/DNN_2/14.PNG)

식에서 보듯 정답이 아닌 나머지 모두 tk는 0이므로 결과에 영향을 주지 않는다.  
따라서 실질적으로 정답일 때의 추정의 자연로그를 계산하는 식이다.

#### 즉, 교차 엔트로피 오차는 정답일 때의 출력이 전체 값을 정하게 된다.

### 3.1.3 왜 손실 함수를 설정하는가?

궁금적인 목적은 높은 '정확도'를 끌어내는 매개변수(가중치) 값을 찾는 것이다.  
그렇다면 정확도라는 지표를 놔두고 손실 함수의 값이라는 방법을 택하는 이유는 뭘까?

답은 **미분**의 역할 때문이다. 신경망 학습에서는 손실함수의 미분을 계산하고, 미분 값이 0이 되는  
쪽으로 먜개변수를 갱신해준다.

### 3.1.4 경사하강법(Gradient Descent)

경사하강법은 함수 값이 낮아지는 방향으로 독립 변수(가중치)값을 변형시켜가면서 최종적으로는  
최소 함수 값을 갖도록 하는 독립 변수 값을 찾는 방법이다.

#### 즉, 가중치가 커질 수록 Loss값이 커지는 중이라면(즉, 기울기의 부호가 양수) 음의 방향으로  
#### 가중치를 옮기고, 그 반대이면 양의 방향으로 옮기면 된다.

### 3.1.5 신경망에서의 기울기

신경망 학습에서도 기울기를 구해야 한다. 기울기는 가중치 매개변수에 대한 손실함수의 기울기이다.

![15](/assets/images/DNN_2/15.PNG)

3.2 신경망 학습 절차
---

신경망 학습의 절차는 다음과 같다.

1. 미니배치 : 학습데이터를 하나씩 학습하는게 아닌 batch_size로 묶어서 학습시켜 효율 극대화
2. 기울기 산출 : Loss function을 이용해 Loss의 기울기를 산출
3. 매개변수 갱신 : 경사하강법으로 기울기를 줄여나가면서 매개변수 갱신
4. 반복 : 1 ~ 3 단계 반복 학습

3.3 정리
---

- 신경망 학습은 손실 함수를 지표로, 손실 함수의 값이 작아지는 방향으로 가중치 매개변수를 갱신한다.
- 가중치 매개변수를 갱신할 때는 가중치 매개변수의 기울기를 이용하고, 기울어진 방향으로 가중치의 값을 갱신하는 작업을 반복한다.
- 미분을 이용해 가중치 매개변수의 기울기를 구할 수 있다.
