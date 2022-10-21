from . import (
    Bert,
    Roberta,
    XLNet,
)


class AutoLanguageModel:

    LANGUAGE_MODEL_MAPPING_NAMES = {
        "bert": Bert,
        "bert-base": Bert,
        "bert-large": Bert,
        "roberta": Roberta,
        "roberta-base": Roberta,
        "roberta-large": Roberta,
        "xlnet": XLNet,
    }

    @classmethod
    def get_class_name(cls, lm_name):
        return cls.LANGUAGE_MODEL_MAPPING_NAMES[lm_name]
