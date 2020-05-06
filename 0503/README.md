# 0503_conference

## 팀회의록

### 예측모델을 위한 변수선택

- 기술통계조사 자료의 데이터 추출: `X 변수`

- 기업실태 조사의 `영업이익성장률`  :  `Y 변수` 

  - 18 개 따로 모델링 -> 상위 5개 모델링
    - 변수 선택 법(?)

- 변수 선택한것

  ```python
  # X_list
  index_list = [x2, mainincm, sep1, sep2, sep3, sep4, n3, n4, sep6, sep6n, sep7, sep8, sep9, sep10, c1s0, CN1, CN2,  
  ```

  - 대소문자 주의하기 (아마 다 대문자로 바꿔야할지도)

    - 함수 써서 대문자로 바꾸자

    `---? 이걸왜하지? 재무재표를보면되는데?`

----------------------------------

## IDEA

1. 영업이익성장률로 기업선정
2. 경기전망data이용 -> 지원효과 잘나오는지
3. 기업선정 X 
   1. 18가지로 분류
   2. `현 지원정책 세분화`
4. `워드 클라우드 ->> 중소기업의 문제점 배경정도로 쓰기`
   - 문제점:
     - 퍼주기식의 지원
     - 진짜 원하는것이 뭔지 알고 지원해줘야 좋다(돈낭비X)

## 해야할것

- 규민
  - 정부의 중소기업 지원 기사 wordcloud 분석
- 인영
  - 결과물 초안 작성
- 동석
  - 18개 군집 특성 세분화 (조금더 자세히)