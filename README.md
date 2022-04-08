# Bert-hate-comment-detector
- Bert기반 악성 댓글, 성차별/혐오 댓글 탐지 모델
> 소셜미디어, 온라인 상 활동 중 악성 댓글 혹은 메세지는 큰 사회적 문제입니다.
> 이러한 악성 댓글을 정확하게 찾아내기 위해서는 문맥을 파악할 수 있어야합니다.
> Transformer 기반의 pre-trained Bert 모델을 사용하여 [Korean-hate-speech](https://github.com/kocohub/korean-hate-speech)데이터를 text-classification학습시킵니다.

## DEMO
- Environment setting
```
conda create -n hate_detector python=3.8
conda activate hate_detector
pip install pandas scikit-learn numpy tensorflow transformer matplotlib koco
```
- Command line execution
```
python run.py
```

- Train model with Python code
```python
from preprocess import Preprocess_with_bert
from run import Detect_hate_comments

# hate => 악성댓글 탐지
# contain_gender_bias => 성혐오/차별 표현 탐지
target = ['hate', 'contain_gender_bias']
model_name = ['bert-base-multilingual-cased', 'kykim/bert-kor-base']
dir_path = ['bert-ckpt', 'kor-bert-ckpt']
epochs = 2
batch_size = 128

detect = Detect_hate_comments(target=target[0], 
                              model_name=model_name[0], 
                              dir_path=dir_path[0])
detect.train_model(epochs=epochs, batch_size=batch_size)
```

- Evaluation
```python
val_acc, val_f1, cls_report = detect.evaluation(batch_size=batch_size)
print('Accuracy : ', val_acc)
print('F1_Score : ', val_f1)
print(cls_report)
```

- Test model with Python code
```python
test_result = detect.test_model(batch_size=batch_size)
print(test_result)
```

- Test model for custom data
```python
sentence = "푸틴한테는 한마디도 못하는 역겨운 거지나라 합정공주년"
print(detect.predict(sentence))
```

## Performance & Evaluation
- 악성 댓글 탐지 모델 평가 비교

|        |Bert-base-multilingual|kyKim/Bert-kor|
|--------|----------------------|--------------|
|Accuracy|0.73|0.78|
|F1-Score|0.80|0.82|
|Recall  |0.79|0.75|
|Precision|0.80|0.90|

- 성차별/혐오 댓글 탐지 모델 평가 비교

|        |Bert-base-multilingual|kyKim/Bert-kor|
|--------|----------------------|--------------|
|Accuracy|0.92|0.94|
|F1-Score|0.69|0.75|
|Recall  |0.61|0.66|
|Precision|0.80|0.88|

- 악성 댓글 탐지 with Bert-base-multilingual

```python
sentence = "푸틴한테는 한마디도 못하는 역겨운 거지나라 합정공주년"
print(detect.predict(sentence))
# 1 (악성댓글)

sentence_ = "강아지와 개새끼의 차이가 뭔가요"
print(detect.predict(sentence_))
# 0 (악성댓글 X)
```

![download](https://user-images.githubusercontent.com/61719257/162421710-f7fa71bc-6c31-4727-aa0a-78040f82425b.png)

- 악성 댓글 탐지 with kyKim/Bert-kor

```python
sentence = "푸틴한테는 한마디도 못하는 역겨운 거지나라 합정공주년"
print(detect.predict(sentence))
# 1 (악성댓글)

sentence_ = "강아지와 개새끼의 차이가 뭔가요"
print(detect.predict(sentence_))
# 1 (악성댓글)
```

![download-1](https://user-images.githubusercontent.com/61719257/162421722-f6f18fba-fc7b-411d-8ec0-cfe5f59367d4.png)

- 성차별/혐오 댓글 탐지 with Bert-base-multilingual

```python
sentence2 = "응. 한남충 여기 하나 추가요"
print(detect_gender.predict(sentence2))
# 0 (악성댓글 X)

sentence2_ = "진짜 여가부 폐지할까요?"
print(detect_gender.predict(sentence2_))
# 0 (악성댓글 X)
```

![download-2](https://user-images.githubusercontent.com/61719257/162421727-80cc9604-f0b4-4f30-8328-c05deef1f38a.png)

- 성차별/혐오 댓글 탐지 with kyKim/Bert-kor

```python
sentence2 = "응. 한남충 여기 하나 추가요"
print(detect_gender.predict(sentence2))
# 1 (악성댓글)

sentence2_ = "진짜 여가부 폐지할까요?"
print(detect_gender.predict(sentence2_))
# 0 (악성댓글 X)
```

![download-3](https://user-images.githubusercontent.com/61719257/162421743-d866a93d-99f5-4061-b354-2741a3898768.png)
