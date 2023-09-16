import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline

def main():
    finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone', num_labels=3)
    tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')

    nlp = pipeline("sentiment-analysis", model=finbert, tokenizer=tokenizer)

    while True:
        user_input = input("Sentence (or exit): ")
        
        if user_input.lower() == "exit":
            break
            
        print(nlp(user_input))

if __name__ == "__main__":
    main()
