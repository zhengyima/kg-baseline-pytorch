import torch 
from torch import nn
import numpy as np
#import matplotlib.pyplot as plt
from torch.autograd import Variable


def seq_max_pool(x):
    """seq是[None, seq_len, s_size]的格式，
    mask是[None, seq_len, 1]的格式，先除去mask部分，
    然后再做maxpooling。
    """
    seq, mask = x
    seq = seq - (1 - mask) * 1e10
    return torch.max(seq, 1)

def seq_and_vec(x):
    """seq是[None, seq_len, s_size]的格式，
    vec是[None, v_size]的格式，将vec重复seq_len次，拼到seq上，
    得到[None, seq_len, s_size+v_size]的向量。
    """
    seq , vec  = x
    vec = torch.unsqueeze(vec,1)
    
    vec = torch.zeros_like(seq[:, :, :1]) + vec
    return torch.cat([seq, vec], 2)

def seq_gather(x):
    """seq是[None, seq_len, s_size]的格式，
    idxs是[None, 1]的格式，在seq的第i个序列中选出第idxs[i]个向量，
    最终输出[None, s_size]的向量。
    """
    seq, idxs = x
    batch_idxs = torch.arange(0,seq.size(0)).cuda()

    batch_idxs = torch.unsqueeze(batch_idxs,1)

    idxs = torch.cat([batch_idxs, idxs], 1)

    res = []
    for i in range(idxs.size(0)):
        vec = seq[idxs[i][0],idxs[i][1],:]
        res.append(torch.unsqueeze(vec,0))
    
    res = torch.cat(res)
    return res


class s_model(nn.Module):
    def __init__(self,word_dict_length,word_emb_size,lstm_hidden_size):
        super(s_model,self).__init__()

        self.embeds = nn.Embedding(word_dict_length, word_emb_size).cuda()
        self.fc1_dropout = nn.Sequential(
            nn.Dropout(0.25).cuda(),  # drop 20% of the neuron 
        ).cuda()


        self.lstm1 = nn.LSTM(
            input_size = word_emb_size,
            hidden_size = int(word_emb_size/2),
            num_layers = 1,
            batch_first = True,
            bidirectional = True
        ).cuda()


        self.lstm2 = nn.LSTM(
            input_size = word_emb_size,
            hidden_size = int(word_emb_size/2),
            num_layers = 1,
            batch_first = True,
            bidirectional = True
        ).cuda()

        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=word_emb_size*2, #输入的深度
                out_channels=word_emb_size,#filter 的个数，输出的高度
                kernel_size = 3,#filter的长与宽
                stride=1,#每隔多少步跳一下
                padding=1,#周围围上一圈 if stride= 1, pading=(kernel_size-1)/2
            ).cuda(),
            nn.ReLU().cuda(),
        ).cuda()
        self.fc_ps1 = nn.Sequential(
            nn.Linear(word_emb_size,1),
        ).cuda()

        self.fc_ps2 = nn.Sequential(
            nn.Linear(word_emb_size,1),
        ).cuda()



    def forward(self,t):
        mask = torch.gt(torch.unsqueeze(t,2),0).type(torch.cuda.FloatTensor) #(batch_size,sent_len,1)
        mask.requires_grad = False
  
        outs = self.embeds(t)

        t = outs
        t = self.fc1_dropout(t)

        

        t = t.mul(mask) # (batch_size,sent_len,char_size)

        t, (h_n, c_n) = self.lstm1(t,None)
        t, (h_n, c_n) = self.lstm2(t,None)

        t_max,t_max_index = seq_max_pool([t,mask])
  

        t_dim = list(t.size())[-1]
        h = seq_and_vec([t, t_max])
  

        h = h.permute(0,2,1)
       
        h = self.conv1(h)
    
        h = h.permute(0,2,1)


        ps1 = self.fc_ps1(h)
        ps2 = self.fc_ps2(h)
        

        return [ps1.cuda(),ps2.cuda(),t.cuda(),t_max.cuda(),mask.cuda()]

class po_model(nn.Module):
    def __init__(self,word_dict_length,word_emb_size,lstm_hidden_size,num_classes):
        super(po_model,self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=word_emb_size*4, #输入的深度
                out_channels=word_emb_size,#filter 的个数，输出的高度
                kernel_size = 3,#filter的长与宽
                stride=1,#每隔多少步跳一下
                padding=1,#周围围上一圈 if stride= 1, pading=(kernel_size-1)/2
            ).cuda(),
            nn.ReLU().cuda(),
        ).cuda()

        self.fc_ps1 = nn.Sequential(
            nn.Linear(word_emb_size,num_classes+1).cuda(),
            # nn.Softmax(),
        ).cuda()

        self.fc_ps2 = nn.Sequential(
            nn.Linear(word_emb_size,num_classes+1).cuda(),
            # nn.Softmax(),
        ).cuda()
    
    def forward(self,t,t_max,k1,k2):

        k1 = seq_gather([t,k1])

        k2 = seq_gather([t,k2])

        k = torch.cat([k1,k2],1)
        h = seq_and_vec([t,t_max])
        h = seq_and_vec([h,k])
        h = h.permute(0,2,1)
        h = self.conv1(h)
        h = h.permute(0,2,1)

        po1 = self.fc_ps1(h)
        po2 = self.fc_ps2(h)

        return [po1.cuda(),po2.cuda()]






