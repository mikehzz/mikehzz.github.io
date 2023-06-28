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


### response 객체

웹 브라우저에게 보내는 응답 정보를 담고 있다.

- 헤더 정보 입력
- 리다이렉트 하기

#### 웹 브라우저 캐시 제어

웹 캐시 또는 HTTP 캐시는 서버 지연을 줄이기 위해 웹 페이지, 이미지,  
기타 유형의 웹 멀티미디어 등의 웹문서들을 임시 저장하기 위한 정보기술이다.  
웹 캐시 시스템은 이를 통과하는 문서들의 사본을 저장하며 이후 요청들은 특정 조건을 충족하는 경우  
캐시화가 가능하다.

### out 객체

JSP 페이지가 생성하는 모든 내용은 out 기본 객체를 통해 전송된다.  


```html
<%@ page language="java" contentType="text/html; charset=UTF-8"
    pageEncoding="UTF-8"%>
<%@ taglib  prefix="c" uri="http://java.sun.com/jsp/jstl/core" %>
<%@ include file="no_cache.jsp" %>
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
  <h2>out객체 사용</h2>
  <hr/>
  <%w
    out.print("안녕하세요");//데이터를 출력한다.  
  %>
  <br/>
  out기본객체를 통한 출력
  <%
    out.print("출력한 결과");//데이터를 출력하고, 줄바꿈 문자(\r\n 또는 \n)을 출력
  %>  
</body>
</html>
```

### application 기본 객체

웹 어플리케이션과 관련된 기본 객체  
web.xml

|메서드|반환값|설명|
|----|---|----|
|getInitParameterNames()|Enumeration<String>|설정 파라미터 이름 목록을 리턴|
|getInitParameter(String param|||

```html

<%@page import="java.util.Enumeration"%>
<%@ page language="java" contentType="text/html; charset=UTF-8"
    pageEncoding="UTF-8"%>
<%@ taglib  prefix="c" uri="http://java.sun.com/jsp/jstl/core" %>
<c:set var="CP" value="${pageContext.request.contextPath }"/>  
<%
  Enumeration<String> paramNames = application.getInitParameterNames();
  while(paramNames.hasMoreElements()){
	  
	  String name = paramNames.nextElement();//변수 이름
	  String nameValue = application.getInitParameter(name);
	  out.print("name:"+name+",value="+nameValue+"<br/>");
	  
  }
  
  //서버 정보 읽기
  //getServerInfo() : 서버정보
  String serverInfo = application.getServerInfo();
  out.println(serverInfo+"<br/>");
  //getMajorVersion() : 서블릿의 major버전
  out.println("MajorVersion:"+application.getMajorVersion());
  //getMinorVersion() : 서블릿의 minor버전
  out.println("MinorVersion:"+application.getMinorVersion());
%>
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

</body>
</html>

```

### Cookie

웹 브라우저가 보관하는 데이터이다.(보안에는 취약)  
웹 브라우저는 웹 서버에 요청을 보낼 때 쿠키를 함께 전송한다.

### 쿠키의 구성

이름 : 각각의 쿠키를 구별하는 데 사용되는 이름
값 : 쿠키의 이름과 관련된 값













































