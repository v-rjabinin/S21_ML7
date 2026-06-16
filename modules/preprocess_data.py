from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import pandas as pd


def preprocess_data(path: str, read_json_params: dict, sort_by: list[str], reset_index: bool, to_delete: list[str], cat_feat_to_map: list[str], cat_mappings: list[dict], cat_feat_to_binarize: list[str], categories_to_use: list[tuple[str, ...]]):
    data = pd.read_json(path, **read_json_params)

    for i, feature in enumerate(cat_feat_to_map):
        data[feature] = data[feature].apply(lambda x: cat_mappings[i][x])

    if sort_by != list():
        data.sort_values(by=sort_by, inplace=True, ignore_index=reset_index)
    else:
        data.reset_index(drop=True, inplace=True)

    if to_delete != list():
        data.drop(labels=to_delete, axis="columns", inplace=True)

    dataframes = list()

    for i, feature in enumerate(cat_feat_to_binarize):
        curr_cat = categories_to_use[i]
        curr_cat = set(item.lower() for item in curr_cat)

        data[feature] = data[feature].apply(lambda x: [item.lower() for item in x])
        data[feature] = data[feature].apply(lambda x: [item for item in x if item in curr_cat])

        mlb = MultiLabelBinarizer(classes=list(curr_cat))
        enc_data = mlb.fit_transform(data[feature])
        dataframes.append(pd.DataFrame(enc_data, columns=mlb.classes_))

    if len(cat_feat_to_binarize) != 0:
        other_feat = [col for col in data.columns if not col in cat_feat_to_binarize]
        dataframes.append(data.loc[:, other_feat])

        data = pd.concat(dataframes, axis=1)

    return data


if __name__ == "__main__":
    print("\n\n✅The module containing preprocess_data function was successfully imported✅\n\n")