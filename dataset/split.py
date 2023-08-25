import json
import pandas as pd

def main():

    total_number_of_splits = 3
    
    filename = 'original/News_Category_BUSINESS_Dataset_v3.json'

    dataframes = [pd.DataFrame(), pd.DataFrame(), pd.DataFrame()]    

    with open(filename, 'r') as file:
        counter = 0
        for line in file:
            data = json.loads(line)
            del data['category']
            del data['link']
            df_item = pd.json_normalize(data)
            
            dataframes[counter] = pd.concat([dataframes[counter], df_item], ignore_index=True)
            counter = (counter + 1) % total_number_of_splits

    for index, df in enumerate(dataframes):
        df.to_csv(f'splits/split{index}.csv', index=False)
    

if __name__ == '__main__':
    main()