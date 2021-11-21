import argparse
import numpy as np
import pandas as pd

def create_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-csv', '--csv_file', type=str, required=True,
                        help='The name of the csv file in local directory to convert to text.')
    parser.add_argument('-drop', '--drop_column', nargs='*', type=int, required=False, default=None,
                        help='Indices of columns to disregard from original data.')
    parser.add_argument('-prep', '--preprocess', type=bool, required=False, default=False,
                        help='Indicates whether or not the data should be preprocessed.')
    parser.add_argument('-test', '--test_prop', type=float, required=False, default=None,
                        help='Proportion of data to split off as test data.')
    return(parser)


def preprocess(data):
    data = data.sample(frac=1).reset_index(drop=True)  #shuffles data
    return(data)


def split(data, prop):
    n = data.shape[0]
    subset = np.random.choice(range(n), int(n*prop))
    test_data = data.iloc[subset, :]
    train_indices = np.setdiff1d(range(n), subset)
    train_data = data.iloc[train_indices, :]
    return(train_data, test_data)


def write(data, name):
    with open(name, "w") as f:
        for i in range(data.shape[0]):
            r = list(data.iloc[i,:])
            rs = [str(num) for num in r]
            rt = ' '.join(rs) + "\n"
            f.write(rt)
        f.close()
    return()



def main():
    parser = create_parser()
    args = parser.parse_args()
    df = pd.read_csv(args.csv_file)

    if args.drop_column:
        df.drop(df.columns[args.drop_column], axis=1, inplace=True)

    if args.preprocess:
        df = preprocess(df)

    if args.test_prop:
        train, test = split(df, args.test_prop)
        del df

        out_train = args.csv_file + ".txt_train"
        out_test = args.csv_file + ".txt_test"

        write(train, out_train)
        write(test, out_test)
        return()
        
        
    
    out_name = args.csv_file + ".txt"
    write(df, out_name)

    return()


if __name__ == "__main__":
    main()
