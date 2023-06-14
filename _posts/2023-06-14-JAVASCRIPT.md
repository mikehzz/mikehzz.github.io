---
layout: single
title:  "SPRING"

categories:
  - SPRING
tags:
  - 
  
---
4.2 자바스크립트
---

### 로그인

1. 로그인 화면
2. 로그인 Controller
3. 로그인 Service

아이디 체크

비번 체크

```
      	$.ajax({
    		type: "GET",
    		url:"/ehr/board/board.do",
    		asyn:"true",
    		dataType:"html",
    		data:{
    			work_div:"doInsert",
    			title: $("#title").val(),
    			div: $("#div").val(),
    			reg_id: $("#reg_id").val(),
    			contents: $("#contents").val()	
    		},
    		success:function(data){//통신 성공
        	console.log("success data:"+data);
        },
        error:function(data){//실패시 처리
        	console.log("error:"+data);
        }
    	});
```

전송 type  "GET/POST"  
서버 URL  
비동기 통신 유무  
ajax를 통해 호출한 페이지의 return 형식  
서버로 보낼 데이터














