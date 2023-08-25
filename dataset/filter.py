import json
import pandas as pd

def main():
    """ Filter original dataset by category. """

    target_category = 'BUSINESS'

    with (open('original/News_Category_Dataset_v3.json', 'r') as original,
          open(f'original/News_Category_{target_category}_Dataset_v3.json', 'w') as category_file):
        for line in original:
            line = json.loads(line)
            if line['category'] == target_category: category_file.write(json.dumps(line) + '\n')

if __name__ == "__main__":
    main()