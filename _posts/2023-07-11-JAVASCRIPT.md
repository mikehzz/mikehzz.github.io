---
layout: single
title:  "SPRING"

categories:
  - SPRING
tags:
  - 
  
---
4.1 JS
---

### 즉시 실행 함수
1. 매치검색
```
'/abc/' : "abc" 문자열과 정확히 일치하는 패턴
'/[a-z]/' : 영문 소문자와 일치하는 패턴
'/[0-9]/' : 숫자와 일치하는 패턴
```
2. 수량
```
'/a+/' : a가 1회이상 연속으로 나타나는 패턴
'/a*/' : a가 0회이상 연속으로 나타나는 패턴
'/a?/' : a가 0회 또는 1회 패턴
'/a[3]/' : a가 정확히 3회 연속으로 나타나는 패턴
'/a[3,4]/' : a가 정확히 3~4회 연속으로 나타나는 패턴
```
3. 앵커
```
'/^abc/' : 문자열의 시작이 'abc'로 시작하는 패턴
'/abc$/' : 문자열의 끝이 'abc'로 끝나는 패턴
```

4. 문자열 클래스

```
'/[aeiou]/` : 소문자(a,e,i,o,u)과 매치하는 패턴
'/[0-9a-zA-Z]/' : 숫자 영문 소문자, 영문 대문자로 이루어 진 패턴
```

사용자 ID가 영문자, 숫자, 밑줄로 구성되고  
길이는 5자 이상, 20자 이하여야 한다.

```java
/**
 * 사용자 ID가 영문자, 숫자, 밑줄로 구성되고	
 * 길이는 5자이상, 20이하여야 한다.
 */
'use strict';

function validUserId(userId){
	let pattern = /^[a-zA-Z0-9_]{5,20}$/;
	
	return pattern.test(userId);
}

//테스트
console.log(validUserId('pcwk_java'));//true
console.log(validUserId('pcwk'));//false(5자 이하)
console.log(validUserId('pcwk__1235465578978989456'));//false(20자 초과)
console.log(validUserId('pcwk_.java'));//false(특수문자 사용

console.clear();
//특정 날짜 정규식으로 표현
//YYYY-MM-DD

//let date_pattern = /^[0-9]{4}-[0-9]{2}-[0-9]{2} $/;
let date_pattern = /^d{4}-\d{2}-\d{2}$/;//\d 숫자:[0-9]
let date = '2023-07-11';
 
console.log(date_pattern.test(date));

//핸드폰 정규식 표현 : 3-4-4
let cellphone_pattern = /^\d{3}-\d{3,4}-\d{4}$/;
let phoneNumber = '010-123-4567';

console.log(cellphone_pattern.test(phoneNumber));

```
### Map 객체, Set 객체

Map은 key, value쌍으로 데이터 관리  
Set은 값은 존재하고 중복된 값을 허용하지 않는다.

```java
let sub = new Map();
sub.set("html", 1);//값 추가
sub.set("css", 2);
sub.set("javascript", 3);

//key로 값 조회
console.log(`sub.get('html'):${sub.get('html')}`);
console.log(`sub.get('javascript'):${sub.get('javascript')}`);

for(let key of sub.keys()){
	console.log(`key:${key},value:${sub.get(key)}`);
	
}

//set
let sub01 = new Set();
sub01.add('a');//추가
sub01.add('b');
sub01.add('C');
sub01.add('a');
sub01.add('b');
sub01.add('d');


for(let value of sub01.values()){ //향상된 for
	
	console.log(`value:${value}`);
}

```

### 이벤트

마우스, 키보드, 폼이벤트, 문서로딩 이벤트, 기타 이벤트  

마우스 이벤트

|이벤트|설명|
|----------|------------|
|click|마우스를 클릭 했을 때 이벤트가 발생|
|dbclick|마우스 더블클릭 했을 때 이벤트 발생|
|mousedeover				|마우스를 오버했을 때 이벤트가 발생|
|mouseout				|마우스가 아웃했을 때 이벤트가 발생	|
|mousedown				|마우스를 눌렀을 때 이벤트 발생|
|mouseup				|마우스를 떼었을 때 이벤트 발생|
|mousemove				|	마우스를 움직였을 때|

키이벤트

|이벤트|설명|
|----------|------------|
|keydown|키를 눌렀을 때 이벤트가 발생|
|keyup|키를 떼었을 때 이벤트 발생|
|keypress	|키를 누른 상태에서 이벤트 발생|

폼 이벤트

|이벤트|설명|
|----------|------------|
|focus|포커스가 이동되었을 때 이벤트가 발생|
|blur|포커스가 벗어났을때 이벤트 발생|
|change|값이 변경되었을 때 이벤트 발생|
|submit|submit버튼을 눌렀을때 이벤트가 발생|
|reset|reset버튼을 눌렀을때 이벤트 발생	|
|select	|input, textarea요소 안의 텍스트를 드래그 해서 선택했을 때 이벤트 발생|

로드, 기타 이벤트

|이벤트|설명|
|----------|------------|
|load|로딩이 완료 되었을 때 이벤트 발생		|
|scroll|스크롤바를 움직였을 때 이벤트 발생|
|resize|사이즈가 변경되었을 때|
|abort|이미지 로딩이 중단되었을 때 이벤트 발생|
|load|로딩이 완료 되었을 때 이벤트 발생|
|scroll	|스크롤바를 움직였을 때 이벤트 발생	|
|resize	|사이즈가 변경되었을 때	|
|abort	|이미지 로딩이 중단되었을 때 이벤트 발생|

### 이벤트 연결 방식

인라인, 기본, 표준 이벤트 모델

#### 인라인

html요소에 직접 이벤트를 연결하는 방식  
on + event  
ex)
```
<button onClick='doSave();'>저장</button>
```


### 기본 이벤트 모델

```
html 요소를 취득 한 이후 이벤트를 객체 메서드 형식으로 연결하는 방식.  
ex)  
객체, 메서드 = function(){...}

html 요소를 취득 할 때는 순서상 취득할 요소가 요소 획득 명령어 이전에 있어야 한다.

```

```
//window로딩이 완료 되면
window.onload = function(){
	//button객체 취득
	let btButton = document.getElementById("btn");

	//이벤트 감지 및 처리
	btButton.onclick = function(){
		console.log(`btn click`);
	}	
}

```

```html
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<link rel = "shortcut icon" type="image/x-icon" href="/ehr/favicon.ico">
<title>자바스크립트</title>
<style>
  table{
    border-collapse:collapse;
  }
  
  td{
    padding : 10px;
    text-align: left;
    border : 1px solid blue;
  }
  td.today{
   background-color: #e6f7ff;
  }
  
</style>
<body>
  <h2>자바스크립트</h2>
  <hr/>
  <div class="calendar">
  </div>
  <!-- 
  js04.js:9 Uncaught TypeError: Cannot set properties of null (setting 'onclick')
    at js04.js:9:18
   -->
  <script src="/ehr/resources/js/ed03/js04.js"></script>
  <button id='btn'>기본형</button>
</body>

</head>
</html>
```

### 표준 이벤트 모델

객체.addEventListener('이벤트',함수);의 방식으로 이벤트를 연결


```java
window.onload = function(){
	
	let bt = document.querySelector("#btn");
	
	//event 감지
	bt.addEventListener('click', doSave);//이벤트에 on 붙이지 않음!

	function doSave(){
		alert('click');
	}
	
}
```

```html
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<link rel = "shortcut icon" type="image/x-icon" href="/ehr/favicon.ico">
<title>자바스크립트</title>
<style>
  table{
    border-collapse:collapse;
  }
  
  td{
    padding : 10px;
    text-align: left;
    border : 1px solid blue;
  }
  td.today{
   background-color: #e6f7ff;
  }
  
</style>
<body>
  <h2>자바스크립트</h2>
  <hr/>
  <div class="calendar">
  </div>
  <!-- 
  js04.js:9 Uncaught TypeError: Cannot set properties of null (setting 'onclick')
    at js04.js:9:18
   -->
  <script src="/ehr/resources/js/ed03/js05.js"></script>
  <button id='btn'>기본형</button>
</body>

</head>
</html>
```

### 이벤트 객체

내장 객체 처럼 자바스크립트에서 기본적으로 제공해 주는 객체  

ex)  
```
bt.onclick = function(event){
    event.프로퍼티
    event.메서드
}
```

|프로퍼티|설명|
|-------|----------|
|target|이벤트를 발생시킨 객체를 반환	|
|type|이벤트의 이름을 반환	|
|clientX|이벤트가 발생한 x좌표(브라우저 기준)|
|clientY|이벤트가 발생한 y좌표(브라우저 기준)|
|screenX|이벤트가 발생한 x좌표(모니터 기준)	|
|screenY|이벤트가 발생한 y좌표(모니터 기준)	|
|button|마우스 왼쪽(0), 가운데(1),오른쪽(2)|





### 자바스크립트 selector

요소를 선택하기 위한 선택자(selector)

1. getElementById
```
document.getElementByid() 메서드를 사용하여 id 기반으로 요소 선택
ex) document.getElementById("username");
```

2. getElementByClassName()
```
document.getElementByClassName() 메서드를 사용하여 클래스 기반으로 요소 선택
ex) document.getElementByClassName("usercClass");
```

3. getElementByTagName()
```
document.getElementByTagsName() 메서드를 사용하여 html태그 기반으로 요소 선택
ex) document.getElementByTagName("div");
```
   
4. querySelector() : css선택자를 기반으로 요소 선택
```
document.querySelector("#username") id기반 선택
document.querySelector(".userClass") class기반 선택
```

### 브라우저와 관련된 객체

BOM(Browser Object Model)  
BOM에는 브라우저와 컴퓨터 스크린에 접근할 수 있는 객체의 모음.

#### window객체  
window객체는 웹 브라우저 전반적인 정보 취득이나 제어 등에 관련된 객체

[22]

|객체|설명|
|--------|--------|
|window|bom의 최상위 객체로, 각 프레임별로 하나씩 존재|
|location|현재 URL에 대한 정보	|
|screen|브라우저 외부 환경에 대한 정보	|
|history|브라우저가 접근했던 URL history|
|navigator|브라우저명 버전정보를 속성으로 가지고 있음.|
|document|현재 문서에 대한 정보|

### window open() 메서드

open()메서드는 새로운 윈도우를 만들어 주는 메서드  
window.open("문서주소",윈도이름,옵션=값,...);
window.open("windowopen.html","window팝업",'width=400, height=600,menubar=no, status=no, toolbar=no');

width : 팝업창 가로길이									
height : 팝업창 세로길이									
toolbar=no : 단축도구창(툴바) 표시안함									
menubar=no : 메뉴창(메뉴바) 표시안함									
location=no : 주소창 표시안함									
scrollbars=no : 스크롤바 표시안함									
status=no : 아래 상태바창 표시안함									
resizable=no : 창변형 하지않음									
fullscreen=no : 전체화면 하지않음									
channelmode=yes : F11 키 기능이랑 같음									
left=0 : 왼쪽에 창을 고정(ex. left=30 이런식으로 조절)									
top=0 : 위쪽에 창을 고정(ex. top=100 이런식으로 조절)

```java
let popWindow ;
window.onload = function(){

	const btn = document.querySelector("#btn");
	
	btn.onclick = function(e){
		console.log(`btn.onclick:${e}`);
		//크롬,Edge: resizeable=no, scrollbars=0
		
		//window.open('tmp_menu.html','팝업','width=600, height=400, left=100, top=10, resizeable=no');
		
		//window.open('https://top.cafe.daum.net/pcwk','팝업','width=600, height=400, left=100, top=10');

		//target:
		//_blank : 팝업을 새창에서 연다(default)
		//_self : 현재 페이지에 새창이 열린다.
		//_parent : 부모창에서 팝업이 열린다.
		//_top : 현재 페이지에서 최상의 페이지에 팝업이 열린다.
		popWindow = window.open('tmp_menu.html','_blank','width=600, height=400, left=100, top=10, resizeable=no');
		
	}

}

function closePopup(){
	popWindow.close();//팝업창 닫기
	
}
```

```html
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<link rel="shortcut icon" type="image/x-icon" href="/ehr/favicon.ico" >
<title>자바스크립트</title>
<script src="/ehr/resources/js/util.js"></script>
</head>
<body>
     <h2>window.open():팝업</h2>
     <hr/>
     
     <script src="/ehr/resources/js/ed03/js08.js"></script>
     <button id="btn">팝업</button>
     <button onclick="closePopup();">닫기</button>
</body>
</html>
```

### center popup

1920*1080

[23]

```java
	btn.onclick = function(e){
		console.log(`btn.onclick:${e}`);
		//크롬,Edge: resizeable=no, scrollbars=0
		
		//window.open('tmp_menu.html','팝업','width=600, height=400, left=100, top=10, resizeable=no');
		
		//window.open('https://top.cafe.daum.net/pcwk','팝업','width=600, height=400, left=100, top=10');
		
		//center popup
		let screenWidth = window.screen.width;
		let screenHeight = window.screen.height;
		
		let popupWidth = 600;
		let popupHeight = 400;
		
		let leftValue = (screenWidth/2 - popupWidth/2);
		let topValue = (screenHeight/2 - popupWidth/2);
		console.log(`screenWidth : ${screenWidth}`);
		console.log(`screenHeight : ${screenHeight}`);
		
		console.log(`leftValue : ${leftValue}`);
		console.log(`topValue : ${topValue}`);

		//target:
		//_blank : 팝업을 새창에서 연다(default)
		//_self : 현재 페이지에 새창이 열린다.
		//_parent : 부모창에서 팝업이 열린다.
		//_top : 현재 페이지에서 최상의 페이지에 팝업이 열린다.
		//듀얼모니터는 left값이 달라 가운데 적용 않됨
		popWindow = window.open('tmp_menu.html','_blank','width=600, height=400, left='+leftValue+",top="+340);
		
	}
```

### navigator
브라우저 버전이나 브라우저명등의 정보를 담고 있는 객체

|프로퍼티|설명|
|----|-------|
|프로퍼티|설명|
|프로퍼티|설명|











