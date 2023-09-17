# Sentiment Analysis on Finance News

## /backend

### ./main.py
Loads labelled data (both self-labelled and FinBert-labelled) to train, handled by rf_model.py.

### ./rf_model.py
Loads Scikit-learn's Random Forest classifier to vectorise (using TF-IDF), split, and train each dataframe - also analyses its effectiveness using a confusion matrix, as well as comparing the accuracy of tested data against the model.

### ./bert_labeller.py
Strips labels on the self-labelled dataset and uses HuggingFace API (FinBert model) to label the data, as a means for comparing it to my own Random Forest trained model.

## /dataset

### ./label_engine.py
Script used for easily labelling, and saving raw data.

### ./labelled/split1.csv
The self-labelled dataset used in Random Forest training.

### ./labelled/bert.csv
FinBert-labelled dataset used for comparison.