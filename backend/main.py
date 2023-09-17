import pandas as pd

from rf_model import train_random_forest

def main():
    df = pd.read_csv('./../dataset/labelled/split1.csv')
    df.fillna('', inplace=True)
    df['text'] = df['headline'] + ' ' + df['short_description']

    bert_df = pd.read_csv('./../dataset/labelled/bert.csv')

    print("\nTraining model on self labelled data...")
    train_random_forest(df)

    print("\nTraining model on BERT labelled data...")
    train_random_forest(bert_df)


if __name__ == "__main__":
    main()