---
layout: single
title:  "[딥러닝 1-1]퍼셉트론(Perceptron)"

categories:
  - 딥러닝 이론
tags:
  - 단층 퍼셉트론
  - AND gate
  - NAND gate
  - OR gate
  - 다층 
  - XOR gate
---

**이 글은 밑바닥부터 시작하는 딥러닝 1 책을 참고로 작성**

[밑바닥부터 시작하는 딥러닝 1](https://github.com/WegraLee/deep-learning-from-scratch)

1.1 퍼셉트론 이란?
---

퍼셉트론은 다수의 신호를 입력받아 하나의 신호를 출력하는 것이다.  
퍼셉트론은 1(신호가 흐른다), 0(신호가 흐르지 않는다) 신호를 출력하여 앞으로 전달한다.

![1](/assets/images/Perceptron/1.PNG)

- x1와 x2는 입력 신호
- y는 출력 신호, w는 가중치
- 원은 뉴런 혹은 노드

#### 퍼셉트론의 동작원리

1. 입력 신호가 뉴런에 보내질 때는 각각 고유한 가중치가 곱해진다.(w1x1, w2x2)
2. 뉴런에서 보내온 신호의 총합이 정해진 한계(임계값)을 넘어설 때만 1을 출력한다.

수식은 다음과 같다.

![2](/assets/images/Perceptron/2.PNG)

퍼셉트론은 복수의 입력 신호 각각에 고유한 가중치를 부여한다. 가중치는 각 신호가 결과에 주는 영향력  
을 조절하는 요소로 작용한다. **즉, 가중치가 클수록 해당 신호가 그만큼 더 중요하다는 뜻이다.**

1.2 단순한 논리 회로
---

퍼셉트론에 이용될 수 있는 간단한 논리회로에 대해 설명한다.

### 1.2.1 AND 게이트

AND 게이트는 입력이 2개, 출력은 1개이다. 두 입력이 모두 1일 때만 1을 출력하고,  
그 외에는 0을 출력한다.

![3](/assets/images/Perceptron/3.PNG)

### 1.2.2 NAND 게이트와 OR 게이트

NAND는 Not AND를 의미하며, 그 동작은 AND 게이트의 출력을 뒤집은 것이 된다.

![4](/assets/images/Perceptron/4.PNG)

OR 게이트는 입력 신호 중 하나 이상이 1이면 출력이 1이 되는 논리 회로이다.

![5](/assets/images/Perceptron/5.PNG)

1.3 퍼셉트론 구현하기
---

### 1.3.1 가중치와 편향 구현

퍼셉트론은 입력 신호에 가중치를 곱한 값과 편향을 합해, 그 값이 0을 넘으면 1을 출력하고  
그렇지 않으면 0을 출력한다.

![6](/assets/images/Perceptron/6.PNG)

### 1.3.2 AND함수 구현

```python
def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    tmp = np.sum(w*x) + b
    if tmp <= 0:
    	return 0
    else:
    	return 1
```

### 1.3.3 NAND, OR함수 구현

```python
def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np. array([-0.5, -0.5]) # AND와는 가중치(w와 b)만 다르다.
    b = 0.7
    tmp = np.sum(w * x) + b
    if tmp <= 0:
    	return 0
    else:
    	return 1
    
    
def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    tmp = np.sum(w * x) + b
    if tmp <= 0:
    	return 0
    else:
    	return 1
```

1.4 퍼셉트론의 한계
---

### 1.4.1 XOR 게이트를 구현할 수 없다.

XOR 게이튼는 베타적 논리합이라는 논리 회로이다.

![7](/assets/images/Perceptron/7.PNG)

하지만 단층 퍼셉트론으로는 XOR 게이트를 구현할 수 없다.

![8](/assets/images/Perceptron/8.PNG)

하나의 직선으로 XOR을 나누기란 불가능하다. 따라서 단층 퍼셉트론으로는 비선형 영역을  
분리할 수 없다는 의미가 된다.

### 1.4.2 선형과 비선형

직선 하나로는 XOR 게이트를 구현할 수 없지만, 곡선이라면 구현할 수 있다.

![9](/assets/images/Perceptron/9.PNG)

이와 같이 곡선의 영역을 **비선형 영역**, 직선의 영역을 **선형 영역**이라고 한다.

1.5 다층 퍼셉트론
---

단층 퍼셉트론으로는 XOR 게이트를 표현할 수 없다. 층을 하나 더 쌓아서 다층 퍼셉트론으로 XOR을 구현할 수 있다.

### 1.5.1 XOR 게이트 구현

```python
def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y
    
XOR(0, 0) # 0을 출력
XOR(1, 0) # 1을 출력
XOR(0, 1) # 1을 출력
XOR(1, 1) # 0을 출력
```

![10](/assets/images/Perceptron/10.PNG)

이처럼 층이 여러개인 퍼셉트론을 다층 퍼셉트론이라고 한다.

1.6 정리
---

- 퍼셉트론은 입출력을 갖춘 알고리즘
- 퍼셉트론은 가중치, 편향을 매개변수로 설정함.
- 단층 퍼셉트론은 직선형 영역만 표현할 수 있고, 다층 퍼셉트론은 비선형 영역도 표현할 수 있다.
