---
layout: single
title:  "SPRING"

categories:
  - SPRING
tags:
  - 
  
---
4.1 JSP
---

### 흐름 제어 태그

```<c:if test='조건'></c:if>
   if만 있고 else없음
   <c:if test='true'>...</c:if> true
   <c:if test='some text'>...</c:if>    false
   <c:if test='${expr}'>...</c:if>    false
   <c:if test='<%=expr %'>...</c:if>    false
```

```<c:choose>
  <c:when test='조건'>...</c:when>
  <c:when test='조건'>...</c:when>
  <c:when test='조건'>...</c:when>


  <c:otherwise>...</c:otherwise>
</c:choose>
```

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
	<h2>forEach</h2>
	<hr/>
	<h4>1부터 100까지 홀수 합</h4>
	<c:set var="sum" value="0" />
	<c:forEach var="i" begin="1" end="100" step="2">
		<c:set var="sum" value="${sum+i}"/>
	</c:forEach>
	
	결과 : ${sum }<br/>
	
	<h4>구구단 : 9단</h4>
	<ul>
		<c:forEach var="i" begin="1" end="9" varStatus="status">
			<li>${status.index}-${status.index}&nbsp;&nbsp; 9 * ${i} = ${9 * i}</li>
		</c:forEach>
	</ul>
	
	<h4>Map</h4>
	
	<c:forEach var="i" items="${map}">
	  ${i.key} = ${i.value}<br/>
	</c:forEach>
	
</body>
</html>
```
```<c:forTokens var="token" item="문자열" delims="구분자"> 

</c:forTokens>
```

### URL처리 태그
```html
<c:url>태그

<c:url value="URL" var="varName" scope="영역">
  <c:param name="이름" value="값" />  
</c:url>
```
















