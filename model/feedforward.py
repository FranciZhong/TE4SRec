import torch


class FeedforwardLayer(torch.nn.Module):
    def __init__(self, d_input, d_hidden, d_output, dropout_rate, is_gelu=True):
        super(FeedforwardLayer, self).__init__()

        self.linear1 = torch.nn.Linear(d_input, d_hidden)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.act = torch.nn.GELU() if is_gelu else torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(d_hidden, d_output)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, x):
        x = self.dropout1(self.linear1(x))
        x = self.act(x)
        x = self.dropout2(self.linear2(x))

        return x
