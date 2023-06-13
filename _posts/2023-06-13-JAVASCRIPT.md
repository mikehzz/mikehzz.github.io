---
layout: single
title:  "SPRING"

categories:
  - SPRING
tags:
  - 
  
---
4.1 자바스크립트
---

### 자바스크립트

```html
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>글자색 바꾸기</title>

<style>
  /* 스타일 */
  body { text-align: center}
  
  #heading {
    color:blue;
  }
  
  #text{
    color:gray;
    font-size:15px;
  }
</style>
 

</head>

<body>
  <h2 id="heading">자바스크립트</h2>
  <p id="text">위 텍스트를 클릭해 보세요</p>
</body>
<script>
  /* Event 감지 */
  let heading = document.querySelector("#heading");
  
  heading.onclick = function(){
	  console.log('heading.onclick');
	  heading.style.color = "red";
  }
  
</script>

</html>
```


```html
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>글자색 바꾸기</title>

<style>
  /* 스타일 */
  body { text-align: center}
  
  #heading {
    color:blue;
  }
  
  #text{
    color:gray;
    font-size:15px;
  }
</style>
 

</head>

<body>
  <h2 id="heading">자바스크립트</h2>
  <p id="text">위 텍스트를 클릭해 보세요</p>
</body>

<script src="ehr/resources/pcwk_javascript/js/change-color.js"></script>

</html>
```


















