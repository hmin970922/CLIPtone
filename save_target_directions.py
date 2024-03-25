import torch
from criteria import CLIPLoss
import pandas as pd


csv_path = './csv/colornames.csv'

df = pd.read_csv(csv_path)
df = df['name'] + ' photo.'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
clip_models = CLIPLoss(device, clip_model = 'RN50')

source_text = 'Normal photo.'
target_text = df[0]

direction_features = clip_models.compute_text_direction(source_text, target_text)
for idx in range(1, len(df)):
    target_text = df[idx]
    direction_feature = clip_models.compute_text_direction(source_text, target_text)
    direction_features = torch.cat([direction_features, direction_feature], dim=0)
    
    if idx % 1000 == 0:
        print(direction_features.shape)

torch.save(direction_features, './target_direction_RN50.pth')