from dataset_load import PG19_DATASET, subset_text_iterator
from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers


### Tokenizer ###

tokenizer = Tokenizer(models.BPE())
tokenizer.normalizer = normalizers.NFKC()
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
tokenizer.decoder = decoders.ByteLevel()
trainer = trainers.BpeTrainer(
    vocab_size=50000,
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    special_tokens=["<PAD>", "<BOS>", "<EOS>"],
)

### Train ###


it = subset_text_iterator(PG19_DATASET, column_name="text", indices=1)
tokenizer.train_from_iterator(it, trainer=trainer)
tokenizer.save("transformer/data/tokenizer-model.json")