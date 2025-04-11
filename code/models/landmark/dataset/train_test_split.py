import pandas as pd
from typing import Dict


def train_test_split(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    # should already be sorted, but just in case
    df = df.sort_values(["label", "data_source"]).reset_index(drop=True)

    # 6 letters, 1 will be assigned to each of the 6 videos for each label
    letters = ["A", "B", "C", "D", "E", "F"]

    # Initialize an empty list to store the split groups
    dataset_split_groups = []

    # Iterate over each group of labels
    # (not changing anything in the df here, just looping to match the length)
    for label, group in df.groupby("label"):
        # Shift the letters along by moving the first letter to the end ABCDEF -> BCDAEF
        letters = letters[1:] + letters[:1]
        # Append the split group letters to the list
        dataset_split_groups.extend(letters)

    # Add the split groups to the DataFrame
    df["dataset_split_group"] = dataset_split_groups

    # This dict will use 3/6 for train, 2/6 for val, 1/6 for test
    # If want to change the split, or which videos are used for each split, we can change this dict
    dataset_split_dict = {
        "A": "train",
        "B": "train",
        "C": "train",
        "D": "trian",
        "E": "val",
        "F": "test",
    }

    # Map the split groups to the dataset splits
    df["dataset_split"] = df["dataset_split_group"].map(dataset_split_dict)
    return df.drop(columns=["dataset_split_group"])
