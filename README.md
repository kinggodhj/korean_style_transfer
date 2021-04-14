# korean_style_transfer

## 한국어 스타일 변환 (긍정 <-> 부정)


#### 데이터셋 준비
##### 1. 사용 데이터

한국어 영화 리뷰 https://github.com/e9t/nsmc


##### 2.데이터 토큰화
Google sentencePiece (version 0.1.85)

e.g) '이런 영화는 그만' -> [79, 189, 1188, 4]

./data/sentimental/

```
python data_token.py
```
