import pandas as pd

def main():
    """ Split the unlabelled dataset so multiple people can help label it. """

    id = 0
    filename = f'splits/split{id}.csv'
    data = pd.read_csv(filename)

    if data.empty:
        print('There are no rows to label.')
        return

    labelled_data = pd.DataFrame()
    data = data.T
    counter = 0
    while not data.empty:
        row = data.pop(counter)
        print(f'headline:\t\t{row["headline"]}')
        print(f'short description:\t{row["short_description"]}')

        label = None
        while label not in ['-1', '0', '1', 'exit']:
            label = input(f'Label (-1, 0, 1, exit): ')
        if label == 'exit': break

        row['label'] = label
        row = pd.DataFrame(row).T
        labelled_data = pd.concat([labelled_data, row])

        counter += 1

    # add the popped row back to the dataframe
    data = data.join(row)
    data = data.T
    data.to_csv(f'splits/split{id}.csv', mode='w', header=True, index=False)

    # append the labelled data to the labelled data store
    flag = False
    try:
        pd.read_csv(f'labelled/split{id}.csv')
    except FileNotFoundError:
        flag = True
    labelled_data.to_csv(f'labelled/split{id}.csv', mode='a', header=flag, index=False)


if __name__ == '__main__':
    main()
