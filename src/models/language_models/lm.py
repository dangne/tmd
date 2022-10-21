from abc import abstractmethod


class LanguageModel():
    '''
    def __init__(self, tokenizer=None, reconstructor=None, rec_kwargs=None, *args, **kwargs):
        super(LanguageModel, self).__init__(*args, **kwargs)
        self.tokenizer = tokenizer  
        self.reconstructor = reconstructor
        self.rec_kwargs = rec_kwargs
    '''

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def set_reconstructor(self, reconstructor, tmd_layer=-1, **kwargs):
        self.reconstructor = reconstructor
        self.tmd_layer = tmd_layer
        self.rec_kwargs = kwargs

    @abstractmethod
    def text2token(self, texts):
        # Return a list of tokenized sentences
        pass

    @abstractmethod
    def token2emb(self, tokens):
        # Return the sentence embeddings of a list of sentences
        pass

    @abstractmethod
    def text2emb(self, texts):
        # Return the sentence embeddings of a list of sentences
        pass

    @abstractmethod
    def classify_embs(self, embs):
        # Classify a given list of embeddings
        pass

    @abstractmethod
    def classify_texts(self, texts):
        # Classify a given list of sentences
        pass

    def forward(self, x):
        raise NotImplementedError

    @property
    def emb_dim(self):
        raise NotImplementedError
