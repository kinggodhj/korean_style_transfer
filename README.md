# korean_style_transfer

## 한국어 스타일 변환


#### 스타일의 정의 
부정 <-> 긍정

#### 데이터셋 준비
##### 1.데이터 토큰화
Google sentencePiece (version 0.1.85)

e.g) '이런 영화는 그만' -> [79, 189, 1188, 4]

./data/sentimental/

'''
python data_token.py
'''
