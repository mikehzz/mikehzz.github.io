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

[깃허브 링크](https://github.com/mikehzz/Computer_Vision/blob/main/pascal_voc_dataset.ipynb)

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

### 2.4.1 OpenCV 특징

OpenCV는 오픈소스 기반의 최고 인기 컴퓨터 비전 라이브러리로 컴퓨터 비전 기능 일반화에 크게 기여했다.

OpenCV 특징
- 인텔이 초기 개발 주도
- 다양한 플랫폼에서 사용 가능
- 방대한 컴퓨터 비전 관련 라이브러리와 손쉬운 인터페이스 제공

### 2.4.2 OpenCV 이미지 로딩

- imread('파일명')를 이용한 이미지 로딩
  - 파일을 읽어 Numpy array로 변환
 
이미지 파일을 OpenCV로 로딩하는 파이썬 코드를 짜면 다음과 같다. 

```python
import cv2
import matplotlib.pyplot as plt

img_array=cv2.imread('파일명')
plt.imshow(img_array)

```

![5](/assets/images/cv-2/5.JPG)

즉, RGB값이 아닌 BGR값으로 자동 반환되기 때문에 RGB값으로 바꿔줘야한다.

```python
import cv2 
import matplotlib.pyplot as plt

bgr_img_array = cv2.imread(‘파일명’)
rgb_img_array = cv2.cvtColor(bgr_img_array, cv2.COLOR_BGR2RGB)

plt.imshow(rgb_img_array)

```

![6](/assets/images/cv-2/6.JPG)

### 2.4.3 OpenCV 영상 처리 개요

#### OpenCV를 활용한 영상 처리

- OpenCV의 VideoCapture 클래스
  - 동영상을 개별 Frame으로 하나씩 읽어들이는 기능 제공
- VideoWriter
  - VideoCapture로 읽어들인 개별 Frame을 동영상 파일로 Write 수행

OpenCV를 활용한 영상처리는 다음과 같이 구현된다.

```python
cap=cv2.VideoCapture(video_input_path) # input (.mp4)

... # 어떠한 과정

vid_writer=cv2.VideoWriter(video_output_path, ...) # output (.mp4)

```

입력영상을 VideoCapture 클래스로 개별 Frame으로 읽어들이고 읽어 들인 Frame을  
어떠한 과정을 통해 Object Detecter된 영상으로 반환한다.

대략적으로 다음과 같이 구현할 수 있다.

```python
# 프레임 하나씩 읽고 쓰기
while True:
	# 다음 프레임 유무, 이미지 프레임
	hasFrame, img_frame=cap.read()
        if not hasFrame:
    	    print('더 이상 처리할 frame이 없습니다')
            break
        
        # 가공할 수 있는 새로운 파일 생성
        vid_writer.write(img_frame)
```

VideoCapture 개요는 다음과 같다.

1. 생성 인자로 입력 video 파일 위치를 받아 생성
```python
cap=cv2.VideoCapture(video_input_path)
```

2. 입력 video 파일의 다양한 속성 가져오기 가능
- 영상 Frame 너비
```python
cap.get(cv2.CAP_PROP_FRAME_WIDTH)
```
- 영상 Frame 높이
```python
cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
```
- 영상 FPS(Frame Per Second)
```python
cap.get(cv2.CAP_PROP_FPS)
```

3. read()는 마지막 Frame까지 차례로 Frame을 읽음

```python
while True:
    hasFrame, img_frame=cap.read() 
    if not hasFrame:
    	print('더 이상 처리할 frame이 없습니다')
        break
```

VideoWriter 개요는 다음과 같다.

- VideoWriter 객체
  - write할 동영상 파일 위치, Encoding 코덱 유형, write fps 수치, frame 크기를 생성자로 입력 받음
  - 이들 값에 따른 동영상 write 수행
  - write 시, 특정 포맷으로 동영상 Encoding 가능
    - DIVX, XVID, MJPG, X264, WMV1, WMV2

```python
cap=cv2.VideoCapture(video_input_path)

codec=cv2.VideoWriter_fourcc(*'XVID')

vid_size=(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

vid_fps=cap.get(cv2.CAP_PROP_FPS)

vid_writer=cv2.VideoWriter(video_output_path,vid_fps,vid_size)

```

기본적으로 다음과 같이 OpenCV에서 영상을 처리한다.

다음 링크를 들어가면 Colab환경에서 OpenCV 영상처리를 구현한 것을 확인 할 수 있다.

[깃허브 링크](https://github.com/mikehzz/Computer_Vision/blob/main/opencv_image_n_video.ipynb)

### 출처 : 인프런 "딥러닝 컴퓨터 비전"  
