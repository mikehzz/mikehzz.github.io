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

익명 함수 중에는 일회성으로 사용되는 함수가 있다.  
즉시 실행 함수는 선언과 동시에 실행되며, 함수 명이 없기 때문에 재호출 할 수 없다.

```
function(){
  실행문
}
```

```
/**
 * 즉시 실행 함수 
 */

function goLife() {
	console.log('즉시 실행 함수!');
}

(function(){
	goLife();
})()

let instance = (function(){
	console.log('즉시 실행 함수2');
})();
```

### 매개 변수

함수를 호출 할 때 전달하는 변수.  
function 함수명(매개변수1, 매개변수2,...){
    실행문;
}

함수명(매개변수1, 매개변수2,...);

let 변수명 = function(매개변수1, 매개변수2,...){


};
변수명(매개변수1, 매개변수2,...);


#### typeScript

[https://www.typescriptlang.org/play?ssl=15&ssc=32&pln=14&pc=1#code/PTAEHUFMBsGMHsC2lQBd5oBYoCoE8AHSAZVgCcBLA1UABWgEM8BzM+AVwDsATAGiwoBnUENANQAd0gAjQRVSQAUCEmYKsTKGYUAbpGF4OY0BoadYKdJMoL+gzAzIoz3UNEiPOofEVKVqAHSKymAAmkYI7NCuqGqcANag8ABmIjQUXrFOKBJMggBcISGgoAC0oACCbvCwDKgU8JkY7p7ehCTkVDQS2E6gnPCxGcwmZqDSTgzxxWWVoASMFmgYkAAeRJTInN3ymj4d-jSCeNsMq-wuoPaOltigAKoASgAywhK7SbGQZIIz5VWCFzSeCrZagNYbChbHaxUDcCjJZLfSDbExIAgUdxkUBIursJzCFJtXydajBYo4ZoeMheRDwPoMYHsGhfNxmZjsBjMSD8WDQdSJRnwPQiLwAIgAoqsGIgFvoxUlsWLwA5UAByYQAOUgEjFQRUAHkvmR3oIeVpIEdUI4FK5pHhQE46Tphlh9CgEIhocJLqyJPToiIDBxsQsmKwODwgop3DQCJqZZB8oJUJROCMALygMUAK0TgjFAG5FAhOIJ4O4AtB4MwABQAA3jifyoAAJABvJvIAC+9YAlIWgA
](https://www.typescriptlang.org/play)

```
let pName:string = "james";
console.log(`pName: ${pName}`);
```
다음을 아래와 같이 변경 해줌.
```
let pName = "james";
console.log(`pName: ${pName}`);
```

### default param

ES6에서 추가

```
/**
 * default param
 */

'use strict';

function showMessage(message, from='unknown'){
	console.log(`message : ${message}`);
	console.log(`from : ${from}`);
}

showMessage('hi', '친구');

function showMessage2(message, fron){
	
	if(from === undefined){
		from = 'unknown';
	}
	
	console.log(`message : ${message}`);
	console.log(`from : ${from}`);
}

//인자 1개만 전달 가능!
showMessage2('hello');

```

### call by reference

* "Call by reference"는 프로그래밍에서 사용되는 매개변수 전달 방식 중 하나입니다.  
* 이 방식에서는 함수에 매개변수로 전달된 인자의 참조(주소)가 전달되어,  
* 함수 내에서 해당 인자를 수정할 수 있습니다.  


```
//객체를 인자로 받는 함수
function modifyObject(obj){
	obj.name = '홍길동'
	obj.age = 22;
	
}

//객체 생성
let person = {
	name: "이상무",
	age : 23
}

console.log(`호출전 : person`);
console.log(`person:${person.name}`);
console.log(`person:${person.age}`);

modifyObject(person);

console.log(`호출후 : person`);
console.log(`person:${person.name}`);
console.log(`person:${person.age}`);

```

### 파라미터 전달 방법

```
function pFunc(a,b,c){
	
	for(let i=0;i<arguments.length;i++){
		console.log(`arguments[${i}]:${arguments[i]}`);

	}
	
}

pFunc(11, 12, 13)

```

### 가변인자

```
function sum(...numbers){
	
	let total = 0;
	for(let number of numbers){
		total += number;
	}
	
	return total;
}

console.log(sum(1,2,3)); //6

console.log(sum(1,2,3,4,5,6,7,8,9,10)); //55

console.log(sum(10)); //10

console.log(sum()); //0

```

### Arrow function

->  
arrow 함수는 function키워드를 생략하고 부등호 => 을 합쳐서 코딩하며 익명함수 형식으로 표현  
단일 명령문일 경우는 함수의 중괄호({})와 return을 생략 가능

```
//일반함수
let doMultiple = function(s1, s2){

  return s1 + s2;
}
```
  
```
//Arrow function
let doMultiple = (s1, s2) => s1 + s2;
```

```
function doMulti(s1,s2){
  let avg = (s1+s2)/2;

  return avg;
}
```

```
let doMulti = (s1,s2) => {
  let avg = (s1+s2)/2;

  return doMulti;
}
```

```
const doAdd = (s1,s2) => (s1+s2);

console.log(`(s1,s2) => (s1+s2) : ${doAdd(12,14)}`);

const doMulti = (s1,s2)=>{
	let mul = s1*s2;
	
	return mul;
};

console.log(`doMulti(12,14):${doMulti(12,14)}`);
```

### 내장 함수

자바스크립트에 내장되어 있는 함수

https://developer.mozilla.org/ko/docs/Web/JavaScript

인코딩, 디코딩  
encodeURLComponent() : 영문, 숫자, (),-,.,~,*,!을 제외한 문자를 인코딩  
dcodeURLComponent() : encodeURLComponent()의 디코딩 함수

숫자, 유/무한 값 판별

isNaN() : 숫자이면 true, 그렇지 않으면 false  
isFinite() : 유한이면 true, 그렇지 않으면 false

숫자, 문자 변환 함수  
Number() :  숫자로 변환해 주는 함수  
paseInt() : 문자를 숫자로 변환  
parseFloat() : 문자를 실수로 변환  
String() : 문자로 변환

문자를 자바스크립트 코드로 변경 함수  
eval()

### 자바스크립트 객체(object)

변수는 데이터 값을 하나 밖에 저장하지 못하지만, 객체는 데이터 값을 필요한 대로 만들어 사용할 수 있다.  
객체의 데이터는  '이름:값' 쌍으로 구성된다.

```
ex)

let 변수이름 = {
  name : "이상무",
  age : 22,
  nationality : '대한민국'

  printOut:function(){},
  
};

```

### class

E56에서 추가  
객체생성, 상속

```
ex)
class 클래스명{
  
  constructor(매개변수1, 매개변수2){
    this.변수01 = 매개변수1;
    this.변수02 = 매개변수2;
  }

  매개변수(){}
  get 메서드(){}
  set 메서드(){}
}

let 변수01 = new 클래스명(매개변수1, 매개변수2);
let 변수02 = new 클래스명(매개변수1, 매개변수2);
```


```
class User{
	
	constructor(name, passwd, age)
	{
		this.name = name;
		this.passwd = passwd;
		this.age = age;
	}
	
	//getter
	get getAge(){
		return this.age;
	}
	
	//setter
	set setAge(value){
		if(value < 0){
			value = 0;
		}
		
		this.age = value;
	}
}

let user01 = new User('PCWK', '4321',22);
console.log(`user01.name:${user01.name}`);

//setter값 전달
user01.setAge = -1;
//getter값 받기
console.log(`user01.age:${user01.getAge}`);
```

### 상속

```class Child extends Parent{
  
}

```

[20]

### callback function

자바스크립트 함수는 일급객체  
함수의 파람으로 함수가 전달 된다.

일급객체(first-class object)란  
변수나 데이터에 할당 할수 있어야 한다.  
객체의 인자로 넘길 수 있어야 한다.  
객체의 리턴값으로 리턴 할수 있어야 한다.  

### 내장 객체

Number, String, Array, Math, Date, RegExp, Map, Set

```
Number
let num = new Number(12);
let num02 = 14;
```

[21]

```
let num = new Number(328.575);

//num.toFixed(1) : 328.6
console.log(`num.toFixed(1) : ${num.toFixed(1)}`);

//num.toFixed(2) : 328.57
console.log(`num.toFixed(2) : ${num.toFixed(2)}`);

//329
console.log(`num.toFixed() : ${num.toFixed()}`);

let num02 = 12;// 십진수 12 -> 1100

console.log(`12의 2진수:${num02.toString(2)}`);//12의 2진수 : 1100
console.log(`12의 16진수:${num02.toString(16)}`);//12의 16진수 : c

//소수점 계산 : 오차
let n01 = 46000;
let n02 = 0.7
//예산 : 32200
console.log(`n01*n02 : ${n01*n02}`);//n01*n02 : 32199.999999999996

//곱해지는 소수가 정수가 나오도록 소수의 자리수를 곱한 뒤 소수 자리수로 다시 나눈다.
console.log(`n01*(n02*10)/10 : ${n01*(n02*10)/10}`);

```

### string 객체

```
let str = new String('자바스크립트');
let str02 = '자바스크립트';

String 객체의 lenght 문자열의 개수를 return
String 주요 메서드

```

### 정규표현식

let reg = new RegExp('Java Script');































