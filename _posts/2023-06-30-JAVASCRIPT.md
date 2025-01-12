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

### 숫자 및 날짜 포맷팅 처리 태그

```
<%@taglib prefix="fmt" uri="http://java.sun.com/jsp/jstl/fmt" %>
```

숫자 ```<fmt:formatNumber>, <fmt:parseNumber>```

날짜 ```<>,<>```

```html
<%@ page language="java" contentType="text/html; charset=UTF-8"
    pageEncoding="UTF-8"%>
<%@taglib prefix="c" uri="http://java.sun.com/jsp/jstl/core" %>
<%@taglib prefix="fmt" uri="http://java.sun.com/jsp/jstl/fmt" %>
<c:set var="CP" value="${pageContext.request.contextPath }"></c:set>
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
  <h2>numberFormat사용법</h2>
  <hr/>
  <c:set var="price" value="10000"/>
  
  통화 :<fmt:formatNumber value="${price}" type="currency" currencySymbol="원"/><br/>
 
  퍼센트 : <fmt:formatNumber value="${price}" type="percent" groupingUsed="false"/><br/>
  <fmt:formatNumber value="${price}" type="number" value="numberType"/>
 숫자 : ${numberType}<br/>

패턴 : <fmt:formatNumber value="${price}" pattern="0000000.00"/>

</body>
</html>

```

```html
<%@ page language="java" contentType="text/html; charset=UTF-8"
    pageEncoding="UTF-8"%>
<%@taglib prefix="c" uri="http://java.sun.com/jsp/jstl/core" %>
<%@taglib prefix="fmt" uri="http://java.sun.com/jsp/jstl/fmt" %>
<c:set var="CP" value="${pageContext.request.contextPath }"></c:set>
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
    <h2>formatDate</h2>
    <hr/>
    <c:set var="now" value="<%=new java.util.Date() %>"/>
    
    full:<fmt:formatDate value="${now}" type="date" dateStyle="full"/><br/>
    short:<fmt:formatDate value="${now}" type="date" dateStyle="short"/><br/>
    
    time:<fmt:formatDate value="${now}" type="time"/><br/>
    both:<fmt:formatDate value="${now}" type="both" dateStyle="full" timeStyle="full"/><br/>

</body>
</html>
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
<!--Load the AJAX API-->
<script  src="https://www.gstatic.com/charts/loader.js"></script>
<script >
   //the corechart package 로딩
   google.charts.load('current', {'packages':['corechart']});
   
   //Callback 함수 지정
   google.charts.setOnLoadCallback(drawChart);
   
   //Callback 
   function drawChart(){
        //차트 데이터 : backend에서 수신
        let data = new google.visualization.DataTable();
        data.addColumn('string', 'Topping');
        data.addColumn('number', 'Slices');
        data.addRows([
          ['Mushrooms', 3],
          ['Onions', 1],
          ['Olives', 1],
          ['Zucchini', 1],
          ['Pepperoni', 2]
        ]);
     
     // chart option: 제목, 차트 크기(json)
       let options = {'title':'How Much Pizza I Ate Last Night',
               'width':800,
               'height':600};
     
       let chart = new google.visualization.PieChart(document.getElementById('chart_div'));
       chart.draw(data, options);
   
   }
   
   
</script>
    


<title>google pie chart</title>
</head>
<body>
     <h2>google pie chart</h2>
     <hr/>
     
     <div id="chart_div"></div>
</body>
</html>
```

/user/levelPerMemberCount.jsp
















