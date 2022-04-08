import pandas as pd
import numpy as np
from konlpy.tag import Mecab
import re
import koco
import tensorflow as tf
from tqdm import tqdm
from transformers import BertTokenizer

tf.random.set_seed(33)
np.random.seed(33)

class Preprocess_with_bert:
    
    def __init__(self, 
                 target=None,
                 pretrained_model_name_or_path='bert-base-multilingual-cased',
                 cache_dir='bert-ckpt',
                ):
        """
        target : ['hate', 'contain_gender_bias']
        bert_tokenizer : ['bert-base-multilingual-cased', 'kykim/bert-kor-base']
        cache_dir : ['bert-ckpt', 'kor-bert-ckpt']
        """
        train_dev = koco.load_dataset('korean-hate-speech', mode='train_dev')
        test = pd.DataFrame(koco.load_dataset('korean-hate-speech', mode='test'))
        train, val = pd.DataFrame(train_dev['train']), pd.DataFrame(train_dev['dev'])
        
        self.MAX_LENGTH = 45
        self.target = target
        
        self.mecab = Mecab()
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path, cache_dir=cache_dir, do_lower_case=False)
        
        self.Train = self.preprocessing(train, mode='train')
        self.Val = self.preprocessing(val, mode='train')
        self.Test = self.preprocessing(test)
        
        self.Train_inputs, self.Train_labels = self.get_inputs_labels_from_bert_tokenizer(self.Train)
        self.Val_inputs, self.Val_labels = self.get_inputs_labels_from_bert_tokenizer(self.Val)       
        self.Test_inputs = self.get_inputs_labels_from_bert_tokenizer(self.Test, mode='test')
        
    def join_sequence(self, x):
        return ' '.join([word for word in x if len(word) > 1])
    
    def preprocessing(self, df, mode=None):
        """
        mode : ['train']
        """
        df['texts'] = df['news_title'] + df['comments']
        df['texts'] = df['texts'].apply(lambda x: x.lower()).apply(lambda x: re.sub(r"[^ ㄱ-ㅣ가-힣+0-9a-z ]", "", x))
        df['texts'] = df['texts'].apply(lambda x: self.mecab.morphs(x))
        df['texts'] = df['texts'].apply(self.join_sequence)

        if mode == 'train':
            if self.target == 'hate':
                df['hate'].where(df['hate']=='none', 1, inplace=True)
                df['hate'].where(~(df['hate']=='none'), 0, inplace=True)
            elif self.target == 'contain_gender_bias':
                df['contain_gender_bias'].where(df['contain_gender_bias']==False, 1, inplace=True)
                df['contain_gender_bias'].where(~(df['contain_gender_bias']==False), 0, inplace=True)        

            return df[['texts', self.target]]

        return df
    
    def bert_tokenizer(self, sentence):
        encoded_dict = self.tokenizer.encode_plus(
            text = sentence,
            add_special_tokens = True,
            max_length = self.MAX_LENGTH,
            padding = 'max_length',
            return_attention_mask = True,
            truncation = True,
        )
        input_id = encoded_dict['input_ids']
        attention_mask = encoded_dict['attention_mask']
        token_type_id = encoded_dict['token_type_ids']
        return input_id, attention_mask, token_type_id
    
    def get_inputs_labels_from_bert_tokenizer(self, df, mode='train'):
        """
        mode : ['train', 'test']
        """
        input_ids = []
        attention_masks = []
        token_type_ids = []

        if mode == 'train':
            data = zip(df['texts'], df['hate'])
            labels = []
            for sentence, label in tqdm(data, total=len(df)):
                try:
                    input_id, attention_mask, token_type_id = self.bert_tokenizer(sentence)

                    input_ids.append(input_id)
                    attention_masks.append(attention_mask)
                    token_type_ids.append(token_type_id)
                    labels.append(label)
                except Exception as e:
                    print(e)
                    pass
            input_ids = np.array(input_ids, dtype=int)
            attention_masks = np.array(attention_masks, dtype=int)
            token_type_ids = np.array(token_type_ids, dtype=int)
            Inputs = (input_ids, attention_masks, token_type_ids)
            Labels = np.asarray(labels, dtype=np.int32)
            return Inputs, Labels

        elif mode == 'test':
            data = df['texts']
            for sentence in tqdm(data, total=len(df)):
                try:
                    input_id, attention_mask, token_type_id = self.bert_tokenizer(sentence)

                    input_ids.append(input_id)
                    attention_masks.append(attention_mask)
                    token_type_ids.append(token_type_id)
                except Exception as e:
                    print(e)
                    pass
            input_ids = np.array(input_ids, dtype=int)
            attention_masks = np.array(attention_masks, dtype=int)
            token_type_ids = np.array(token_type_ids, dtype=int)
            Inputs = (input_ids, attention_masks, token_type_ids)
            return Inputs
        