import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import pandas as pd

class TransformerBiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads, num_layers):
        super().__init__()
        self.embedding = nn.Linear(input_size, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x1 = self.embedding(x)
        x2 = self.transformer_encoder(x1.unsqueeze(1))
        x3, _ = self.lstm(x2)
        x4 = self.fc(x3[:, -1, :])
        return x1, x2, x3, x4

scaler = StandardScaler()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerBiLSTM(input_size=8, hidden_size=216, num_heads=4, num_layers=1).to(device)
model.load_state_dict(torch.load('pipeline.pth'))
model.eval()

df = pd.read_excel('INPUTDATA.xlsx')

data = df.iloc[1:, 1:9].values

targets1 = df.iloc[1:, 9].values

scaler = StandardScaler()
data_normalized = scaler.fit_transform(data)

X_new_tensor = torch.tensor(data_normalized, dtype=torch.float32).to(device)
with torch.no_grad():
    _, _, _, predictions = model(X_new_tensor)

predictions_np = predictions.cpu().numpy().flatten()
print("Predicted:", predictions_np)



