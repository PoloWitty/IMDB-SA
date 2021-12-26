from dataset import Dataset
from datasets import load_metric
from model import Model
import os,json
from tokenizers import Tokenizer
import torch
from tqdm.auto import tqdm
import logging
from torch.utils.tensorboard import SummaryWriter


#load arguments from config.json
config_file = 'test/imdb/rnn/config.json'
with open(config_file,'r') as fp:
    args=json.load(fp) 

# logging setting
logging.basicConfig(level = logging.DEBUG,format = '%(filename)s - %(message)s',filename=os.path.join(args['log_dir'],'result.log'),filemode='w')
logger = logging.getLogger(__name__)

# get dataloader
tokenizer=Tokenizer.from_file(args['tokenizer_path'])
dataset=Dataset(tokenizer)
train_dataloader,test_dataloader=dataset.get_train_test_dataloader(batch_size=args['batch_size'])

#get the model
model = Model(args['vocab_size'],args['emb_size'],args['hid_size'],args['dropout'],args['layer_norm_eps'])
optimizer = torch.optim.Adam(model.parameters())
loss_fn = torch.nn.CrossEntropyLoss()
device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# progress bar
num_training_steps = args['epoch_num']*len(train_dataloader)
progress_bar = tqdm(range(num_training_steps),desc='total training progress',total=num_training_steps)


def train_one_epoch(epoch_index,tb_writer):
    running_loss=0.0
    last_loss=0.0
    for i,batch in enumerate(train_dataloader):
        batch = {k:v.to(device) for k,v in batch.items()}#移到device里
        inputs = batch['input_ids']
        labels = batch['labels']
        outputs = model(inputs)
        loss = loss_fn(outputs,labels)
        loss.backward()
        optimizer.step()

        optimizer.zero_grad()

        #gather data and report
        running_loss+=loss.item()
        progress_bar.update(1)
        if i % 500==499:
            last_loss = running_loss/500#loss per batch
            logging.debug('in epoch{}  batch {} loss: {}'.format(epoch_index,i + 1, last_loss))
            tb_x = epoch_index * len(train_dataloader) + i + 1
            tb_writer.add_scalar('loss/train',last_loss,tb_x)
            running_loss=0.0
        
    return last_loss

def test_one_epoch():
    running_vloss=0.0
    metric= load_metric(args['metric_path'])
    precision = load_metric('precision')
    recall = load_metric('recall')
    for i,batch in tqdm(enumerate(test_dataloader),desc='eval on testset',total=len(test_dataloader)):
        batch = {k:v.to(device) for k,v in batch.items()}
        inputs = batch['input_ids']
        labels = batch['labels']        
        with torch.no_grad():
            outputs = model(inputs)
            vloss=loss_fn(outputs,labels)
            running_vloss +=vloss
            logits = outputs
            predictions = torch.argmax(logits,dim=-1)
            metric.add_batch(predictions=predictions,references=batch['labels'])
            precision.add_batch(predictions=predictions,references=batch['labels'])
            recall.add_batch(predictions=predictions,references=batch['labels'])
    m=metric.compute()
    p = precision.compute()
    r = recall.compute()
    avg_vloss = running_vloss/(i+1)
    
    return avg_vloss,m,p,r

if __name__ == "__main__":
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/imdb_trainer_rnn_{}'.format(timestamp))
    best_accuracy=0

    for epoch in range(args['epoch_num']):
        print('in epoch {}'.format(epoch))
        print()
        # model.train()
        # avg_loss = train_one_epoch(epoch,writer)

        model.load_state_dict(torch.load(args['model_path']+'/checkpoint-2')) # for eval debug only
        model.eval()
        avg_vloss,m,p,r = test_one_epoch()

        logging.debug('result in epoch {}'.format(epoch))
        logging.debug('avg train loss: {}, avg test loss {}'.format(avg_loss,avg_vloss))
        logging.debug('test accuracy:{}'.format(str(m)))
        writer.add_scalars('Training V.S. Testing Loss',{'Training':avg_loss,'Testing':avg_vloss},epoch+1)
        writer.flush()

        if m['accuracy'] > best_accuracy:
            best_accuracy = m['accuracy']
            torch.save(model.state_dict(),os.path.join(args['model_path'],'checkpoint-'+str(epoch)))

    print('Finish fine-tuning')
