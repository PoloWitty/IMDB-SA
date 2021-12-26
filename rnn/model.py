import torch
from torch import nn

class Model(nn.Module):
    def __init__(self,vocab_size,emb_size,hid_size,dropout,layer_norm_eps):
        super(Model,self).__init__()
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.hid_size = hid_size
        self.dropout = dropout
        self.embedding = nn.Embedding(self.vocab_size,self.emb_size,padding_idx=0)
        self.lstm = nn.LSTM(input_size=self.emb_size,hidden_size=self.hid_size,num_layers = 2,batch_first=True,bidirectional =True,dropout=self.dropout) #两层双向lstm加dropout
        self.dp = nn.Dropout(self.dropout)
        self.fc1 = nn.Linear(self.hid_size*2,self.hid_size)
        self.fc2 = nn.Linear(self.hid_size*3,self.hid_size*3)#残差连接层
        self.layerNorm = nn.LayerNorm(self.hid_size*3,eps=layer_norm_eps)
        self.fc3 = nn.Linear(self.hid_size*3,self.hid_size)
        self.fc4 = nn.Linear(self.hid_size,2)
        self.act_fn = nn.ReLU()

    def forward(self,x):
        '''
        input: (batch,seq_len)
        output: (batch,2)
        '''
        x = self.embedding(x) # (batch,seq_len,emb_isze)
        x = self.dp(x) 
        x,_ = self.lstm(x) # (batch,seq_len,2*hidden_size)
        x = self.act_fn(self.fc1(x)) # (batch,seq_len,hid_size) #把前向和后向的信息合在一起, 当做这个单词的embedding
        avg_x = nn.AvgPool2d((x.shape[1],1))(x).squeeze() #(batch,hid_size)
        max_x = nn.MaxPool2d((x.shape[1],1))(x).squeeze() #(batch,hid_size)
        seq_x = torch.concat([avg_x,max_x,x[:,-1,:]],dim=1) # 把平均池化, 最大值池化, 和最后一个单词的隐藏层状态都拼接起来, 结果维度是(batch,hid_size*3)
        x = self.fc2(seq_x)# (batch,hid_size*3)
        x = self.dp(x)
        x = self.layerNorm(x+seq_x)#加上一点点的残差连接, 把句子的信息整合到一起
        x = self.act_fn(self.fc3(x))# (batch,hid_size)
        x = self.fc4(x) # (batch,2)
        return x

if __name__=="__main__":
    x=torch.LongTensor(torch.randint(0,2500,(8,32)))
    print(x.shape)
    model=Model(25000,512,256,0.3,1e-4)
    output = model(x)
    print(output)