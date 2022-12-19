from typing import List, Tuple

from args import ProgramArgs
from utils.mask import mask_sentence
from utils.safer import WordSubstitude

TEXTFOOLER_SET = set(
                ["a", "about", "above", "across", "after", "afterwards", "again", "against", "ain", "all", "almost",
                 "alone", "along", "already", "also", "although", "am", "among", "amongst", "an", "and", "another",
                 "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are", "aren", "aren't", "around", "as",
                 "at", "back", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides",
                 "between", "beyond", "both", "but", "by", "can", "cannot", "could", "couldn", "couldn't", "d", "didn",
                 "didn't", "doesn", "doesn't", "don", "don't", "down", "due", "during", "either", "else", "elsewhere",
                 "empty", "enough", "even", "ever", "everyone", "everything", "everywhere", "except", "first", "for",
                 "former", "formerly", "from", "hadn", "hadn't", "hasn", "hasn't", "haven", "haven't", "he", "hence",
                 "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his",
                 "how", "however", "hundred", "i", "if", "in", "indeed", "into", "is", "isn", "isn't", "it", "it's",
                 "its", "itself", "just", "latter", "latterly", "least", "ll", "may", "me", "meanwhile", "mightn",
                 "mightn't", "mine", "more", "moreover", "most", "mostly", "must", "mustn", "mustn't", "my", "myself",
                 "namely", "needn", "needn't", "neither", "never", "nevertheless", "next", "no", "nobody", "none",
                 "noone", "nor", "not", "nothing", "now", "nowhere", "o", "of", "off", "on", "once", "one", "only",
                 "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "per",
                 "please", "s", "same", "shan", "shan't", "she", "she's", "should've", "shouldn", "shouldn't",
                 "somehow", "something", "sometime", "somewhere", "such", "t", "than", "that", "that'll", "the",
                 "their", "theirs", "them", "themselves", "then", "thence", "there", "thereafter", "thereby",
                 "therefore", "therein", "thereupon", "these", "they", "this", "those", "through", "throughout", "thru",
                 "thus", "to", "too", "toward", "towards", "under", "unless", "until", "up", "upon", "used", "ve",
                 "was", "wasn", "wasn't", "we", "were", "weren", "weren't", "what", "whatever", "when", "whence",
                 "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether",
                 "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "with", "within",
                 "without", "won", "won't", "would", "wouldn", "wouldn't", "y", "yet", "you", "you'd", "you'll",
                 "you're", "you've", "your", "yours", "yourself", "yourselves"]
            )

class Augmentor(object):
    def __init__(self, args: ProgramArgs):
        self.training_type = args.training_type
        self.mask_token = '[MASK]'
        self.safer_aug_set = f'{args.workspace}/cache/embed/{args.dataset_name}/perturbation_constraint_pca0.8_100.pkl'
        self.aug = WordSubstitude(self.safer_aug_set)

    def augment(self, sentence: str, n: int) -> List[str]:
        if self.training_type == 'mask':
            return mask_sentence(sentence, 0.7, self.mask_token, n)
        elif self.training_type == 'dne':
            return [sentence] * n
        elif self.training_type == 'safer':
            return self.aug.get_perturbed_batch(sentence.lower(), rep=n)
        else:
            return [sentence]


