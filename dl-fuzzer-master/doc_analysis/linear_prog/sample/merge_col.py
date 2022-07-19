import pandas as pd

def main(csv_path, save_path, col1, col2, new_col):
    df = pd.read_csv(csv_path)
    df['tmp'] = df[[col1, col2]].apply(lambda x: ';'.join(x.dropna()), axis=1)
    df['tmp'] = df['tmp'].apply(lambda x: ' '.join(pd.unique(x.split(';'))))
    # .apply(lambda x: ';'.join(list(set(x))), axis=1)
    df = df.drop([col1, col2], axis=1)
    df = df.rename(columns={"tmp": new_col})
    df.to_csv(save_path)


main('./tf_label30.csv', './tf30_merged.csv', 'structure', 'tensor_t', 'structure')
main('./pt_label30.csv', './pt30_merged.csv', 'structure', 'tensor_t', 'structure')
main('./mx_label30.csv', './mx30_merged.csv', 'structure', 'tensor_t', 'structure')