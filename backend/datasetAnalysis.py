import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score


finance_news = pd.read_csv('dataset/labelled/split1.csv', encoding='ISO-8859-2')
finance_news.head()

finance_news['text'] = finance_news['headline'] + ' ' + finance_news['short_description']

label_mapping = {-1: 'negative', 0: 'neutral', 1: 'positive'}
finance_news['label'] = finance_news['label'].map(label_mapping)

finance_news.dropna(subset=['text', 'label'], inplace=True)

X = finance_news['text'].tolist()
y = finance_news['label'].tolist()

finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone', num_labels=3)
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')

labels = {0: 'neutral', 1: 'positive', 2: 'negative'}

sent_val = list()
for x in X:
    inputs = tokenizer(x, return_tensors="pt", padding=True)
    outputs = finbert(**inputs)[0]

    val = labels[np.argmax(outputs.detach().numpy())]
    print(x, '----', val)
    print('##################################################')
    sent_val.append(val)

print("LENGTH: ", len(sent_val))
accuracy = accuracy_score(y, sent_val)
print(f'Accuracy: {accuracy:.2f}')