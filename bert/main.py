from datasets import load_dataset,load_metric
from tokenizers import Tokenizer
from transformers import AutoModelForSequenceClassification
from transformers import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import os,json
from dataset import Dataset
from torch.utils.tensorboard import SummaryWriter, writer
from datetime  import datetime
import logging

config_file = 'test/imdb/bert/config.json'
#load arguments from config.json
with open(config_file,'r') as fp:
    args=json.load(fp) 

logging.basicConfig(level = logging.DEBUG,format = '%(filename)s - %(message)s',filename=os.path.join(args['log_dir'],'result.log'),filemode='w')
logger = logging.getLogger(__name__)

os.environ['HF_DATASETS_OFFLINE'] = '1'
raw_data = load_dataset("imdb")

tokenizer = Tokenizer.from_file(args['tokenizer_path'])
dataset=Dataset(tokenizer,raw_data)
train_dataloader,test_dataloader = dataset.get_train_test_dataloader(batch_size=args['batch_size'])

model = AutoModelForSequenceClassification.from_pretrained(args['pretrained_model'], num_labels=2)


optimizer = AdamW(model.parameters(), lr=5e-5)

num_epochs = args['epoch_num']
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)

device = torch.device("cuda:2") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

progress_bar = tqdm(range(num_training_steps))

# model.train()
# for epoch in range(num_epochs):
#     for batch in train_dataloader:
#         batch = {k: v.to(device) for k, v in batch.items()}
#         outputs = model(**batch)
#         loss = outputs.loss
#         loss.backward()

#         optimizer.step()
#         lr_scheduler.step()
#         optimizer.zero_grad()
#         progress_bar.update(1)

def train_one_epoch(epoch_index,tb_writer):
    running_loss=0.0
    last_loss=0.0
    
    for i,batch in enumerate(train_dataloader):
        batch = {k:v.to(device) for k,v in batch.items()}#移到device里
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        optimizer.zero_grad()

        #gather data and report
        running_loss+=loss.item()
        progress_bar.update(1)
        if i % 1000==999:
            last_loss = running_loss/1000#loss per batch
            print('in epoch{}  batch {} loss: {}'.format(epoch_index,i + 1, last_loss))
            logging.debug('in epoch{}  batch {} loss: {}'.format(epoch_index,i + 1, last_loss))
            tb_x = epoch_index * len(train_dataloader) + i + 1
            tb_writer.add_scalar('loss/train',last_loss,tb_x)
            running_loss=0.0
        
    return last_loss

def test_one_epoch():
    running_vloss=0.0
    metric= load_metric("accuracy")
    precision = load_metric('precision')
    recall = load_metric('recall')
    for i,batch in tqdm(enumerate(test_dataloader),desc='eval on testset',total=len(test_dataloader)):
        batch = {k:v.to(device) for k,v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
            vloss=outputs.loss
            running_vloss +=vloss
            logits = outputs.logits
            predictions = torch.argmax(logits,dim=-1)
            metric.add_batch(predictions=predictions,references=batch['labels'])
            precision.add_batch(predictions=predictions,references=batch['labels'])
            recall.add_batch(predictions=predictions,references=batch['labels'])
    m=metric.compute()
    p = precision.compute()
    r = recall.compute()
    avg_vloss = running_vloss/(i+1)
    
    return avg_vloss,m,p,r


timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/imdb_trainer_{}'.format(timestamp))
best_accuracy=0

model.eval()
avg_vloss,m,p,r = test_one_epoch()
logging.debug('At first: avg test loss {}'.format(avg_vloss))
logging.debug('test accuracy:{}, precision: {}, recall: {}'.format(str(m),str(p),str(r)))

for epoch in range(args['epoch_num']):
    model.train()
    avg_loss = train_one_epoch(epoch,writer)

    # model.load_state_dict(torch.load('test/imdb/bert/checkpoint_without_lr_schedule/checkpoint1')) # for eval debug only
    model.eval()
    avg_vloss,m,p,r = test_one_epoch()
    logging.debug('avg train loss: {}, avg test loss {}'.format(avg_loss,avg_vloss))
    logging.debug('test accuracy:{}, precision: {}, recall: {}'.format(str(m),str(p),str(r)))
    writer.add_scalars('Training V.S. Testing Loss',{'Training':avg_loss,'Testing':avg_vloss},epoch+1)
    writer.flush()

    if m['accuracy']> best_accuracy:
        best_accuracy = m['accuracy']
        torch.save(model.state_dict(),os.path.join(args['model_path'],'checkpoint'+str(epoch)))

print('Finish fine-tuning')
