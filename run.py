import os
import numpy as np
import pandas as pd
import tensorflow as tf
from transformers import BertModel, TFBertModel
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt

from preprocess import Preprocess_with_bert

tf.random.set_seed(33)
np.random.seed(33)

BATCH_SIZE = 32
NUM_EPOCHS = 2

class TFBertClassifier(tf.keras.Model):
    def __init__(self, model_name, dir_path, num_class):
        super(TFBertClassifier, self).__init__()
        self.bert = TFBertModel.from_pretrained(model_name, cache_dir=dir_path)
        self.dropout = tf.keras.layers.Dropout(self.bert.config.hidden_dropout_prob)
        self.classifier = tf.keras.layers.Dense(num_class,
                                                kernel_initializer=tf.keras.initializers.TruncatedNormal(self.bert.config.initializer_range),
                                                name='classifier')

    def call(self, inputs, attention_mask=None, token_type_ids=None, training=False):
        outputs = self.bert(inputs, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output, training=training)
        logits = self.classifier(pooled_output)

        return logits


class Detect_hate_comments:
    def __init__(self, target='hate', model_name=None, dir_path=None):
        """
        target : ['hate', 'contain_gender_bias']
        model_name : ['bert-base-multilingual-cased', 'kykim/bert-kor-base']
        dir_path : ['bert-ckpt', 'kor-bert-ckpt']
        """
        
        
        self.target = target
        self.model_name = model_name
        self.dir_path = dir_path
        
        self.bert_tokens = Preprocess_with_bert(target=self.target, 
                                                pretrained_model_name_or_path=self.model_name,
                                                cache_dir=self.dir_path)
        
        self.Train_inputs, self.Train_labels = self.bert_tokens.Train_inputs, self.bert_tokens.Train_labels
        self.Val_inputs, self.Val_labels = self.bert_tokens.Val_inputs, self.bert_tokens.Val_labels
        self.Test_inputs = self.bert_tokens.Test_inputs
        
        self.model = TFBertClassifier(model_name=self.model_name,
                                      dir_path=self.dir_path,
                                      num_class=2)
        self.optimizer = tf.keras.optimizers.Adam(3e-5)
        self.loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
        self.checkpoint_path = os.path.join(f'{self.target}/{self.model_name.replce('/','-')}/cp.ckpt')
        self.checkpoint_dir = os.path.dirname(self.checkpoint_path)
        self.cp_callback = ModelCheckpoint(self.checkpoint_path, 
                                           monitor='val_accuracy', 
                                           verbose=1,
                                           save_best_only=True, 
                                           save_weights_only=True)
        try:
            self.model.load_weights(self.checkpoint_path)
            print("PRE-TRAINED MODEL LOAD COMPLETES!")
        except:
            pass
        
    def train_model(self, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE):
        self.model.compile(optimizer=self.optimizer,
                           loss=self.loss,
                           metrics=[self.metric])

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.history = self.model.fit(self.Train_inputs, 
                                      self.Train_labels,
                                      epochs=epochs, 
                                      batch_size=batch_size, 
                                      validation_data = (self.Val_inputs, self.Val_labels),
                                      callbacks=[self.cp_callback])
        self.model.load_weights(self.checkpoint_path)
    
    def evaluation(self, batch_size=BATCH_SIZE):
        preds = self.model.predict(self.Val_inputs, batch_size=batch_size)
        pred_labels = np.argmax(preds, axis=1)
        val_acc = accuracy_score(self.Val_labels, pred_labels)
        val_f1 = f1_score(self.Val_labels, pred_labels)
        cls_report = classification_report(self.Val_labels, pred_labels)
        return val_acc, val_f1, cls_report
        
    def test_model(self, batch_size=BATCH_SIZE):
        pred = self.model.predict(self.Test_inputs, batch_size=batch_size)
        pred_label = np.argmax(pred, axis=1)
        result_df = self.bert_tokens.Test
        result_df[f'{self.target}_prediction'] = pred_label
        return result_df[['comments', f'{self.target}_prediction']]
    
    def predict(self, sentence):
        input_id, attention_mask, token_type_id = self.bert_tokens.bert_tokenizer(sentence)
        Inputs = (np.array([input_id], dtype=int), np.array([attention_mask], dtype=int), np.array([token_type_id], dtype=int))
        pred = self.model.predict(Inputs)
        pred_label = np.argmax(pred, axis=1)
        return pred_label
        
    def plot_history(self):
        plt.figure(figsize=(10, 5))

        plt.subplot(121)
        plt.plot(self.history.history['loss'], 'b', label='train')
        plt.plot(self.history.history['val_loss'], 'r', label='val')
        plt.title('Loss')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.legend(loc='upper left')

        plt.subplot(122)
        plt.plot(self.history.history['accuracy'], 'b', label='train')
        plt.plot(self.history.history['val_accuracy'], 'r', label='val')
        plt.title('Accuracy')
        plt.xlabel('epochs')
        plt.ylabel('accuracy')
        plt.legend(loc='upper left')

        plt.show();
        

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')
    
    while True:
        target_input = input("[SELECT TARGET FOR DETECTION]\n[1] hate\t[2]gender bias\n")
        if target_input == '1':
            target = 'hate'
            break
        elif target_input == '2':
            target = 'contain_gender_bias'
            break
        else:
            print("WRONG NUMBER!")
    while True:
        model_name_input = input("[SELECT MODEL]\n[1] bert-base-multilingual-cased\t[2]kykim/bert-kor-base\n")
        if model_name_input == '1':
            model_name = 'bert-base-multilingual-cased'
            dir_path = 'bert-ckpt'
            break
        elif model_name_input == '2':
            model_name = 'kykim/bert-kor-base'
            dir_path = 'kor-bert-ckpt'
            break
        else:
            print("WRONG NUMBER!")
    detect = Detect_hate_comments(target=target, model_name=model_name, dir_path=dir_path)
    sentence = input("ENTER YOUR SENTENCE : ")
    result = detect.predict(sentence)
    if result == 1:
        print("Umm...It can hurt others!")
    else:
        print("It seems fine!")
        