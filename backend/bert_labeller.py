import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification

def main():
    df = pd.read_csv('./../dataset/labelled/split1.csv')
    df.fillna('', inplace=True)
    df['text'] = df['headline'] + ' ' + df['short_description']

    # Drop rows with missing data
    df.dropna(subset=['text'], inplace=True)

    X = df['text'].tolist()

    # Load the Hugging Face FinBERT model
    finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone', num_labels=3)
    tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')

    # Encode labels back to (-1, 0, 1)
    label_mapping = {0: -1, 1: 0, 2: 1}

    new_df = pd.DataFrame(columns=['text', 'label'])
    for x in X:
        inputs = tokenizer(x, return_tensors="pt", padding=True)
        outputs = finbert(**inputs)[0]
    
        val = label_mapping[np.argmax(outputs.detach().numpy())]

        new_df = pd.concat([new_df, pd.DataFrame.from_dict({'text': [x], 'label': [val]})], ignore_index=True)

    new_df.to_csv('./../dataset/labelled/bert.csv', index=False)

if __name__ == '__main__':
    main()