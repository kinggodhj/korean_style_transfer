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
모델 설명 (model.py)

1. Transforemr 기본 모델 
 
   Layer 개수: 2 
   모델 사이즈, embedding 사이즈, latent 사이즈: 256

3. Encoder의 output latent vector와의 유사도 곱을 통한 embedding 모듈
 
   Embedding 사이즈: 256
 
main.py 파일에서 모델 구조 생성 및 학습 진행 

```
python main.py
```

