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
#### 1. 모델 설명 (model.py)

1. Transforemr 모델 
 
   Layer 개수: 2 
   모델 사이즈, embedding 사이즈, latent 사이즈: 256

3. 스타일 embedding 모듈
 
   Embedding 사이즈: 256
   
#### 2. 학습 구조 

1. Transforemr 모델 
 
   Encoder의 latent vector와 스타일 embedding vector를 이용하여 
   
   입력 문장을 복원하는 auto-encoder 형태로 학습 진행 (classification loss로 학습 진행)

2. 스타일 embedding 모듈
   
   Transformer의 encoder output인 latent vector와 행렬 곱을 통해 유사도 얻음
   
   유사도를 통해 긍정, 부정을 분류 (classification loss로 학습 진행) 
   

main.py 파일에서 모델 구조 생성 및 학습 진행 

```
python main.py
```

Output: ./save/epoch_ae_model_params.pkl

------------

### 스타일 변형하기

유사도 곱을 통해 학습된 스타일 embedding을 조절하면서 문장의 스타일 변형을 진행

Input: Test 문장

--weight: 스타일 변형 정도를 제어하는 hyper-parameter (0<=w 인 실수이고 0이면 문장 복원의 기능을 가짐)
```
python generation.py --weight 2.0
```
Output: 스타일 변형된 문장


예시

```
Input(positive): 이렇게 좋은데 평점이 왜이러지 ....

Transferred (w=3): 이렇게 별로임 제작... 이게 영화냐...

Transferred (w=2): 이렇게 좋은데 뭘 비추겠지 .... 0

```
