from torch.utils.data import DataLoader

class Dataset(object):
    def __init__(self,tokenizer,raw_data):
        self.tokenizer=tokenizer
        self.raw_data=raw_data
    
    def _tokenize_function(self,examples):
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
        self.tokenizer.enable_truncation(512)
        be=self.tokenizer.encode_batch(examples['text'])
        dic={'input_ids':[],'token_type_ids':[],'attention_mask':[]}
        for e in be:
            dic['input_ids'].append(e.ids)
            dic['token_type_ids'].append(e.type_ids)
            dic['attention_mask'].append(e.attention_mask)
        return dic
    
    def get_tokenized_dataset(self):
        print('Tokenizing...')
        tokenized_dataset=self.raw_data.map(self._tokenize_function,batched=True)
        tokenized_dataset = tokenized_dataset.remove_columns(['text'])
        tokenized_dataset = tokenized_dataset.rename_column('label','labels')
        tokenized_dataset.set_format('torch')
        return tokenized_dataset

    def get_train_test_dataloader(self,batch_size):
        tokenized_dataset = self.get_tokenized_dataset()
        # small_train_dataset = tokenized_dataset['train'].select(range(1000))
        # small_test_dataset = tokenized_dataset['test'].select(range(1000))
        full_train_dataset = tokenized_dataset['train']
        full_test_dataset = tokenized_dataset['test']
        train_dataloader = DataLoader(full_train_dataset,shuffle=True,batch_size=batch_size)
        test_dataloader = DataLoader(full_test_dataset,shuffle=False,batch_size=batch_size)
        return train_dataloader,test_dataloader

if __name__=='__main__':
    import os
    from datasets import load_dataset
    os.environ['HF_DATASETS_OFFLINE'] = '1'
    raw_data = load_dataset('imdb')
    
    from tokenizers import Tokenizer
    tokenizer=Tokenizer.from_file('MRC_adv/ptm/bert-base-uncased/tokenizer.json')
    
    dataset=Dataset(tokenizer,raw_data)
    train_dataloader,test_dataloader=dataset.get_train_test_dataloader(batch_size=8)

    print(next(iter(train_dataloader)))
    print(next(iter(test_dataloader)))
    count = [0, 0]
    for post_batch in train_dataloader:
        print(post_batch["input_ids"].shape)
        print(post_batch["token_type_ids"].shape)
        print(post_batch["attention_mask"].shape)
        print(post_batch["labels"].shape, post_batch["label"])

        print()

        count[0] += len(post_batch["input_ids"])
        count[1] += sum(post_batch["labels"])

    print(count)