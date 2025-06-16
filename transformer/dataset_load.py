from datasets import load_dataset, DatasetDict
from typing import Final, Iterable, Union
from itertools import islice

### Datasets ###

PG19_DATASET: Final[DatasetDict] = load_dataset("deepmind/pg19", split="train", streaming=True, trust_remote_code=True)

### Iterators ### 

def subset_text_iterator(
    dataset,
    column_name: str,
    indices: Union[int, Iterable[int]],
):
    """
    Yield text from `column_name` but only for the first N examples
    (or explicit indices) of a streaming IterableDataset.
    """
    # if the user passed a single int N, we just take the first N
    if isinstance(indices, int):
        # islice(dataset, N) will give us the first N examples
        for example in islice(dataset, indices):
            yield example[column_name]
    else:
        # if they passed an iterable of indices, we materialize up to max(idx)+1
        max_idx = max(indices)
        for i, example in enumerate(islice(dataset, max_idx + 1)):
            if i in indices:
                yield example[column_name]


if __name__ == "__main__":
    with open("transformer/data/KJV.txt", "w") as f:
        for i, line in enumerate(PG19_DATASET.select_columns("text")):
            f.write(line["text"])
            f.write("\n")
            if i > 1:
                break