import torch
from criteria import CLIPLoss
import pandas as pd

device = 'cuda' if torch.cuda.is_available() else 'cpu'
clip_model = 'RN50'
clip_models = CLIPLoss(device, clip_model = clip_model)

csv_path = './csv/colornames.csv'

df = pd.read_csv(csv_path)
df = df['name'] + ' photo.'

feature_list = []

source_text = 'Normal photo.'
for idx in range(len(df)):
    target_text = df[idx]
    direction_feature = clip_models.compute_text_direction(source_text, target_text)
    feature_list.append(direction_feature)
    
    if idx % 1000 == 0:
        print(idx)

direction_features = torch.cat(feature_list, dim=0)
torch.save(direction_features, f'./csv/target_direction_{clip_model}.pth')
print(direction_features.shape)