---
layout: single
title:  "[딥러닝 1-2]신경망(Neural Network)"

categories:
  - 딥러닝 이론
tags:
  - 신경망
  - 활성화 함수
  - 선형
  - 비선형
---

**이 글은 밑바닥부터 시작하는 딥러닝 1 책을 참고로 작성**

[밑바닥부터 시작하는 딥러닝 1](https://github.com/WegraLee/deep-learning-from-scratch)

2.1 퍼셉트론에서 신경망으로
---

퍼셉트론에서는 AND, OR 게이트의 진리표를 보면서 가중치를 사람이 정했다.
신경망은 가중치 매개변수의 적절한 값을 데이터로부터 **자동으로** 학습할 수 있다. 

### 2.1.1 신경망의 예

![1](/assets/images/DNN_2/1.PNG)

신경망을 그림과 같이 표현했다.

입력층, 은닉층, 출력층으로 나뉘어져 있다.  
또한 각 화살표에는 가중치가 있다.

2.2 활성화 함수
---

### 2.2.1 활성화 함수의 역할

활성화 함수는 입력 신호의 총합을 출력 신호로 변환하는 함수이며,  
입력 신호의 총합이 활성화를 일으키는지를 정하는 역할을 한다.

![2](/assets/images/DNN_2/2.PNG)

### 2.2.2 시그모이드 함수

다음은 시그모이드 함수(sigmoid function)를 나타낸 식이다.

![3](/assets/images/DNN_2/3.PNG)

신경망에서는 활성화 함수로 시그모이드 함수를 이용해 신호를 변환하고, 그 변환된 신호를  
다음 뉴런에 전달한다.  
퍼셉트론과 신경망의 주된 차이는 이 활성화 함수 뿐이다.

```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)

plt.plot(x, y)
plt.ylim(-0.1, 1.1) # y축 범위 지정
plt.show()
```

![4](/assets/images/DNN_2/4.PNG)

### 2.2.3 계단 함수

계단 함수는 입력이 0을 넘으면 1을 출력하고, 그 외에는 0을 출력하는 함수이다.

```python
def step_function(x):
    y = x > 0
    return y.astype(np.int)

x = np.arange(-5.0, 5.0, 0.1)
y = step_function(x)

plt.plot(x, y)
plt.ylim(-0.1, 1.1) # y축의 범위 지정
plt.show()
```

![5](/assets/images/DNN_2/5.PNG)

2.3 비선형 함수
---

신경망에서는 활성화 함수로 비선형 함수를 사용해야 한다.

### 2.3.1 ReLU 함수

ReLU는 입력이 0을 넘으면 그 입력을 그대로 출력하고, 0이하이면 0을 출력하는 함수이다.

 ```python
def relu(x):
    return np.maximum(0, x)
```

![6](/assets/images/DNN_2/6.PNG)

2.4 3층 신경망 구현
---

![7](/assets/images/DNN_2/7.PNG)

다음 그림과 같이 3층 신경망이 구성되있으며, 구현해보자.

1. 입력층에서 1층으로의 신호 전달

![8](/assets/images/DNN_2/8.PNG)

2. 1층에서 2층으로의 신호 전달

![9](/assets/images/DNN_2/9.PNG)

3. 2층에서 출력층으로의 신호 전달

![10](/assets/images/DNN_2/10.PNG)

2.5 출력층 설계
---

### 2.5.1 항등 함수와 소프트맥스 함수

**항등함수**는 입력을 그대로 출력한다. 입력과 출력이 항상 같다는 뜻의 항등이다.

![11](/assets/images/DNN_2/11.PNG)

한편 분류에서 사용하는 소프트맥스 함수(Softmax Function)의 식은 다음과 같다.

![12](/assets/images/DNN_2/12.PNG)

### 2.5.2 소프트맥스 함수의 특징

```python
a = np.array( [0.3, 2.9, 4.0] )
y = softmax(a) # softmax함수를 구현했다고 가정

print(y)  #[ 0.01821128 0.24519181 0.73659691]

print( np.sum(y) ) # 소프트맥스 함수 출력의 총합은 1 입니다.
```

다음과 같이 소프트맥스 함수의 특징은 다음과 같다.

- 출력 총합이 1이 된다.
이 성질 덕분에 소프트맥스 함수의 출력을 '확률'로 해석할 수 있다. 
- 소프트맥스 함수를 적용해도 각 원소의 대소 관계는 변하지 않는다.
단조 증가 함수이기 때문에 출력값들의 대소관계는 변하지 않는다.

2.6 정리
---

- 신경망은 입력층, 은닉층, 출력층이 있고, 각 층마다 가중치와 편향으로 계산한다.
- 신경망에서는 비선형 함수로 사용한다.
