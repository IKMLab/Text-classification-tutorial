import pandas as pd
from sklearn.model_selection import train_test_split
import torch


def preprocess_agnews(
    data_name: str,
    data_type: str = "train",
    use_agnews_title: bool = False,
    train_size: float = 0.8,
    random_state: int = 42,
):
    # Read data
    df = pd.read_csv(f"data/{data_name}/{data_type}.csv")
    df["new_label"] = df["Class Index"] - 1

    if data_type == "train":
        # Split train data into train and validation
        train_df, val_df = train_test_split(
            df, train_size=train_size, random_state=random_state
        )

        train_label = train_df["new_label"].tolist()
        val_label = val_df["new_label"].tolist()

        if use_agnews_title:
            train_text = train_df["Title"] + " " + train_df["Description"]
            val_text = val_df["Title"] + " " + val_df["Description"]
            train_text = train_text.tolist()
            val_text = val_text.tolist()
        else:
            train_text = train_df["Description"].tolist()
            val_text = val_df["Description"].tolist()

        return train_text, train_label, val_text, val_label

    else:
        test_label = df["new_label"].tolist()
        if use_agnews_title:
            test_text = df["Title"] + " " + df["Description"]
            test_text = test_text.tolist()
        else:
            test_text = df["Description"].tolist()

        return test_text, test_label


class AGNewsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


if __name__ == "__main__":
    test_text, test_label = preprocess_agnews("agnews", "test", use_agnews_title=True)
