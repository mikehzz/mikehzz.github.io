---
layout: single
title:  "SPRING"

categories:
  - SPRING
tags:
  - 
  
---
4.2 JSP
---

### JSP

jsp는 웹 애플리케이션 개발을 위한 java기반의 서버사이드 스크립팀 언어이다.

jsp는 동적인 웹페이지를 생성하고, 데이터베이스와 상호작용하며, java코드를 포함하여 웹 애플리케이션을 구축하는데 사용된다.

jsp 라이프 사이
[14]

```html
<%@ page language="java" contentType="text/html; charset=UTF-8"
    pageEncoding="UTF-8"%>
<%@ taglib  prefix="c" uri="http://java.sun.com/jsp/jstl/core" %>
<c:set var="CP" value="${pageContext.request.contextPath }"/>


<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<!-- CSS only -->
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-rbsA2VBKQhggwzxH7pPCaAqO46MgnOM80zW1RWuH61DGLwZJEdK2Kadq2F9CUG65" crossorigin="anonymous">
<!-- JavaScript Bundle with Popper -->
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/js/bootstrap.bundle.min.js" integrity="sha384-kenU1KFdBIe4zVF0s0G1M5b4hcpxyD9F7jL+jjXkk+Q2h455rYXK/7HAuoJl+0I4" crossorigin="anonymous"></script>
<script src="${CP}/resources/js/jquery-3.7.0.js"></script>
<script src="${CP}/resources/js/util.js"></script>
<title>Insert title here</title>
</head>
<body>
  <h2>주석</h2>
  <hr/>
  
  <!-- html 주석  : 클라이언트에게 전달되는 주석-->
  
  <%-- jsp 주석 : 클라이언트로 전달되지 않는 주석 --%>
</body>
</html>

<%
  //java 한줄 주석 : 클라이언트에게 전달되는 주석 -->
  //jsp 주석 : 클라이언트로 전달되지 않는 주석 -
%>

```

### jsp 지시어

```<%@ page 속성 = "속성값" ...%>```

|속성|기본값|설명|
|--------|---------|---------|
|contentType|text/html|jsp생성할 문서의 MIME 타입과 캐릭터 인코딩을 지정한다.|
|import||jsp페이지에서 사용할 자바 클래스를 지정한다.|
|session|||
|errorPage|true||
|isErrorPage|||
|language|false|||
|pageEncoding|||

### 표현식(Epression)

### 선언부

메서드를 작성할 때 사용.

```
<%!
public 리턴타입 메서드이름(파람){


}
%>
```

### include

include지시어는 현재 jsp파일에 다른 html, jsp문서를 포함할 수 있다.
layout에 처리


### 








































