import torch
from torch import nn
import torch.nn.functional as F

class Twolayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        torch.manual_seed(args.seed)
        self.cls = torch.nn.Sequential(
          nn.Linear(28*28, 14*14),
          nn.ReLU(),
          nn.Dropout(),
          nn.Linear(14*14, 10),
          nn.Softmax(dim=1)
        )
        for layer in self.cls:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight.data, 0, 0.01)
                nn.init.constant_(layer.bias.data, 0)

    def forward(self, input):
        output = self.cls(input)
        return output
    
class LogisticRegression(nn.Module):
    def __init__(self, args):
        super().__init__()
        torch.manual_seed(args.seed)
        self.lr = nn.Linear(20, 2)
        nn.init.normal_(self.lr.weight.data, 0, 0.01)
        nn.init.constant_(self.lr.bias.data, 0)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input):
        output = self.lr(input)
        output = self.softmax(output)
        return output

class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return self.softmax(x)


class SentimentLSTM(nn.Module):
    def __init__(self, embedding_dim = 300, hidden_dim = 128, dropout = 0.5, output_dim = 2):
        super(SentimentLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim*2, output_dim)
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, x):
        packed_output, (hidden, cell) = self.lstm(x)
        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        out = self.fc(hidden)
        out = F.softmax(out, dim=1)
        
        return out
    