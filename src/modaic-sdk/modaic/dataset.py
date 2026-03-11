import dspy


class Dataset:
    def __init__(self, data: list):
        self.data = data

    def to_dspy(self) -> list:
        return dspy.Dataset(self.data)

    @classmethod
    def from_csv(cls, file_path: str) -> "Dataset":
        with open(file_path, "r") as file:
            data = file.read()
        return cls(data)

    @classmethod
    def from_hub(cls, dataset_name: str) -> "Dataset":
        from datasets import load_dataset

        data = load_dataset(dataset_name)
        return cls(data)
