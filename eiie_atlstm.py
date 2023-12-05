import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 模型
class EIIE_AT_LSTM(nn.Module):
    def __init__(self, num_features, num_hiddens, num_stocks, device):
        super(EIIE_AT_LSTM, self).__init__()
        self.num_hiddens = num_hiddens
        self.num_stocks = num_stocks
        self.lstm = nn.LSTM(num_features, num_hiddens, batch_first=True)
        self.attention = nn.MultiheadAttention(num_hiddens, 8)
        self.layer_norm1 = nn.LayerNorm(num_hiddens)
        self.layer_norm2 = nn.LayerNorm(num_hiddens)
        self.fc = nn.ModuleList([nn.Linear(num_hiddens, 1) for _ in range(num_stocks)])
        self.device = device

    def forward(self, X):
        # TODO: 优化eiie结构，比较eiie和咱们的atlstm的效果哪个好
        batch_size, num_stocks, seq_length, num_features = X.shape
        outputs = []
        for i in range(num_stocks):
            stock_data = X[:, i, :, :]
            h0 = torch.zeros(1, stock_data.size(0), self.num_hiddens).to(self.device)
            c0 = torch.zeros(1, stock_data.size(0), self.num_hiddens).to(self.device)
            lstm_out, _ = self.lstm(stock_data, (h0, c0))
            normed_lstm_out = self.layer_norm1(lstm_out)
            lstm_out_transposed = normed_lstm_out.transpose(0, 1)
            attention_out, _ = self.attention(lstm_out_transposed, lstm_out_transposed, lstm_out_transposed)
            attention_out_transposed = attention_out.transpose(0, 1)
            combined_out = attention_out_transposed + normed_lstm_out
            normed_combined_out = self.layer_norm2(combined_out)
            aggregated_features = normed_combined_out.sum(dim=1)
            stock_prediction = self.fc[i](aggregated_features)
            outputs.append(stock_prediction)
        outputs = torch.cat(outputs, dim=1)
        return outputs

# 数据处理函数
def create(group, num_timesteps):
    return np.array([group[i:i + num_timesteps].values for i in range(len(group) - num_timesteps + 1)])

def create_labels(data, num_timesteps=550):
    labels_list = []
    for _, group in data.groupby('stock_id'):
        label_samples = create(group[['target']], num_timesteps)
        labels_list.append(label_samples)
    labels_array = np.array(labels_list)
    return labels_array

def create_samples(data, num_timesteps=550, features=None):
    if features is None:
        features = ['seconds_in_bucket', 'imbalance_size', 'imbalance_buy_sell_flag', 
                    'reference_price', 'matched_size', 'far_price', 'near_price', 
                    'bid_price', 'bid_size', 'ask_price', 'ask_size', 'wap']
    samples_list = []
    for _, group in data.groupby('stock_id'):
        group = group[features]
        samples = create(group, num_timesteps)
        samples_list.append(samples)
    samples_array = np.array(samples_list)
    return samples_array

def process_stock_data(data):
    samples = create_samples(data)
    label_samples = create_labels(data)
    return samples, label_samples

# DataLoader创建函数
def create_dataloader(samples, labels, batch_size=32):
    tensor_samples = torch.tensor(samples, dtype=torch.float32)
    tensor_labels = torch.tensor(labels, dtype=torch.float32)
    dataset = TensorDataset(tensor_samples, tensor_labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

# 训练函数
def train_and_save_model(train_features, train_labels, model, save_path, batch_size=32, lr=0.001, num_epochs=50, device='cuda'):
    train_loader = create_dataloader(train_features, train_labels, batch_size)
    loss = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for X, Y in train_loader:
            X, Y = X.to(device), Y.to(device)
            optimizer.zero_grad()
            output = model(X)
            l = loss(output, Y.view(Y.size(0), -1))
            l.backward()
            optimizer.step()
            total_loss += l.item()
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}')
    torch.save(model.state_dict(), save_path)

# 加载
data = pd.read_csv('data/train_cleaned.csv')
data.drop(['Unnamed: 0'], axis=1, inplace=True)

# 模型和数据准备
num_features = 12
num_hiddens = 128
num_stocks = 10
num_timesteps = 550
model = EIIE_AT_LSTM(num_features, num_hiddens, num_stocks, device).to(device)
samples, labels = process_stock_data(data)
train_and_save_model(samples, labels, model, 'model_stock.pth', batch_size=55, lr=0.001, num_epochs=50, device=device)
