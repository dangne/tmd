import torch
from transformers import AutoTokenizer
from models.language_models.lm import LanguageModel


class HuggingFaceLanguageModel(LanguageModel):
    def text2token(self, texts, text_pairs=None):
        tokenized_examples = self.tokenizer(
            texts, 
            text_pairs,
            padding='max_length',
            truncation=True, 
            return_tensors='pt',
        )
        inputs = {k: torch.as_tensor(v, device=self.device) for k, v in tokenized_examples.items()}
        return inputs

    def text2emb(self, texts, text_pairs=None, return_outputs=False):
        inputs = self.text2token(texts, text_pairs)
        return self.token2emb(**inputs, return_outputs=return_outputs)

    def classify_texts(self, texts):
        pooled_output = self.text2emb(texts)
        return self.classify_embs(pooled_output)

    @property
    def emb_dim(self):
        return self.config.hidden_size

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, load_tokenizer=True, *args, **kwargs):
        instance = super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        if load_tokenizer:
            tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
            instance.set_tokenizer(tokenizer)
        return instance
