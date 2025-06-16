from tokenizers import Tokenizer
from typing import Final

TOKENIZER: Final[Tokenizer] = Tokenizer.from_file("transformer/data/tokenizer-model.json")

if __name__ == "__main__":
    print(TOKENIZER.encode("Hello, world!"))