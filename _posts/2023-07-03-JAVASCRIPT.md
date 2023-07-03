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

### Spring scheduler

서비스를 주기적으로 실행해야할 때 사용.  
ex)회원 등업

servlet-context.xml에 task namespace추가 하고 bean을 참조해 Cron 표현식으로 표현

Cron 표현식 - 특수문자

각 특수문자의 의미도 살펴보겠습니다.  
● * : 모든 값을 뜻합니다.  
● ? : 특정한 값이 없음을 뜻합니다.  
● - : 범위를 뜻합니다. (예) 월요일에서 수요일까지는 MON-WED로 표현  
● , : 특별한 값일 때만 동작 (예) 월,수,금 MON,WED,FRI  
● / : 시작시간 / 단위 (예) 0분부터 매 5분 0/5  
● L : 일에서 사용하면 마지막 일, 요일에서는 마지막 요일(토요일)  
● W : 가장 가까운 평일 (예) 15W는 15일에서 가장 가까운 평일 (월 ~ 금)을 찾음  
● # : 몇째주의 무슨 요일을 표현 (예) 3#2 : 2번째주 수요일  

![18](/assets/images/SPRING_1/18.png)


![19](/assets/images/SPRING_1/19.png)


```html
      <!--
                    초 분 시 일 월 요일 
                   
                   초:0-59,-,*,/                
                   분:0-59,-,*,/
                   시:0-23,-,*,/
                   일:1-31,-,*,/,?
                   요일:1-7,SUM-SAT,-,*,/
                   
       * : 향상
       / : 시작시간, 반복 간격
           ex) 분 5/10 -> 5분에  시작해서 10분 간격으로 동작
       ? : 특정 값이 아닌 어떤 값이든 상관 없음
       -->

```

```html
    <!-- scheduler cron -->
    <task:scheduled-tasks>
      <task:scheduled ref="cronTest" method="test" cron="0/10 * * * *?"/>
    </task:scheduled-tasks>
```

```java
@Component
public class CronTest implements PcwkLoger{
	
	public CronTest() {
		LOG.debug("----------------------------");
		LOG.debug("-CronTest()-");
		LOG.debug("----------------------------");
	}
	
	public void test() {
		LOG.debug("----------------------------");
		LOG.debug("-cront test 10초 1회-");
		LOG.debug("----------------------------");	
	}
	
```

@Scheduled  
annotation 사용  

```java

@Component
public class CronTest implements PcwkLoger{
	
	public CronTest() {
		LOG.debug("----------------------------");
		LOG.debug("-CronTest()-");
		LOG.debug("----------------------------");
	}

	@Scheduled(cron = "0/5 * * * * *")
	public void testAno() {
		LOG.debug("----------------------------");
		LOG.debug("-@Scheduled annotation 5초에 1회-");
		LOG.debug("----------------------------");	
		
	}
	
	
}

```

```html
    <task:annotation-driven scheduler="scheduler" />
    <task:scheduler id="scheduler" pool-size="10" />
    
    <!-- scheduler cron -->
    <task:scheduled-tasks>
      <task:scheduled ref="cronTest" method="test" cron="0/10 * * * * ?"/>
      <!-- 10:30 매일, 매월, 매주 -->
      <task:scheduled ref="memberUpgradeLevelCron" method="upgradeLevelsCron" cron="0 33 10 * * ?"/>
    </task:scheduled-tasks>
```

### naver api

naver_search.jsp  
NaverSearchController.java  
NaverSearchService.java  
NaverSerachServiceImpl.java  

### 검색 url https://openapi.naver.com/v1/search/blog.json

















