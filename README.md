# Bert-hate-comment-detector
- Bert기반 악성 댓글, 성차별/혐오 댓글 탐지 모델

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