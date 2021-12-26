# train a tokenizer using imdb dataset from scratch

import os,json
from datasets import load_dataset
from tokenizers import normalizers
from tokenizers.normalizers import BertNormalizer,NFD
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.models import BPE
from tokenizers import Tokenizer
from tokenizers.trainers import BpeTrainer


CONFIG_FILE='test/imdb/rnn/config.json'

def def_tokenizer():
    normalizer = normalizers.Sequence([
        NFD(),
        BertNormalizer(clean_text=True,#去掉控制字符并用classic one替换掉所有的空格
                handle_chinese_chars=True,#在汉字左右插入空格
                strip_accents=True,#去掉所有的注音
                lowercase=True#小写
    )])
    pre_tokenizer = Whitespace()

    tokenizer = Tokenizer(BPE(unk_token='[UNK]'))
    tokenizer.normalizer = normalizer
    tokenizer.pre_tokenizer = pre_tokenizer
    return tokenizer

def batch_iterator(raw_data,batch_size=1000):
    for i in range(0,len(raw_data),batch_size):
        yield raw_data[i:i+batch_size]['text']

def get_args():
    with open(CONFIG_FILE,'r') as fp:
        args=json.load(fp)
    return args

args=get_args()


tokenizer = def_tokenizer()
trainer = BpeTrainer(vocab_size=args['vocab_size'],show_progress=True,min_frequency=50,special_tokens=['[PAD]','[UNK]'])


# load the dataset
os.environ['HF_DATASETS_OFFLINE'] = '1'
raw_data = load_dataset('imdb',split='train+test+unsupervised')

# train process
tokenizer.train_from_iterator(batch_iterator(raw_data),trainer=trainer,length=len(raw_data))

# save the result
tokenizer.save(args['tokenizer_path'])

# How to load
# tokenizer = Tokenizer.from_file(args['tokenizer_path'])