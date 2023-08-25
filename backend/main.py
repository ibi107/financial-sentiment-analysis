import json
import pandas as pd

def main():
    with open('../dataset/News_Category_Dataset_v3.json', 'r') as file:
        json_data = json.load(file)
    data = pd.json_normalize(json_data['data'])

    print(data.head)


if __name__ == "__main__":
    main()