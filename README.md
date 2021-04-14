# korean_style_transfer

## 한국어 문장 스타일 변환 (긍정 <-> 부정)
------------
### 데이터셋 준비
#### 1. 사용 데이터

한국어 영화 리뷰 https://github.com/e9t/nsmc


#### 2. Tokenizer를 이용한 BPE 모델 학습
Google sentencePiece (version 0.1.85)

Training 데이터 셋 (긍정, 부정 모두) 을 이용한 BPE 학습

Input: all.txt 
```
python sentence_piece.py
```
Output: sentimental.model


#### 3.데이터 토큰화 

학습한 BPE 모델 (sentimental.model) 을 이용하여 전체 데이터 셋 Tokenization

```
cd ./data/sentimental/
python data_token.py
```
```
>>>import sentencepiece as spm

>>>sp=spm.SentencePieceProcessor()
>>>sp.Load('sentimental.model')
>>>input='이런 영화는 그만'
>>>sp.EncodeAsIds(input)
>>>[79, 189, 1188, 4]
```

------------

### 학습하기
