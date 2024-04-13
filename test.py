import torch
import time
import argparse
import os
from dataset.dataset import SingleImageDataset, DirectionDataset
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from ailutmodel import AiLUT
from adaptation import AdaptationModule
from criteria import CLIPLoss

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--backbone', type=str, default='tpami')
    parser.add_argument('--backbone_checkpoint_dir', type=str, default='./checkpoint/base_network')
    parser.add_argument('--backbone_checkpoint_name', type=str, default='AiLUT-FiveK-sRGB.pth')
    parser.add_argument('--n_ranks', type=int, default=3)
    parser.add_argument('--n_vertices', type=int, default=33)
    parser.add_argument('--en_adaint', type=bool, default=True)
    parser.add_argument('--en_adaint_share', type=bool, default=False)
    parser.add_argument('--pretrained', type=bool, default=False)
    parser.add_argument('--n_colors', type=int, default=3)
    parser.add_argument('--dataset_dir', type=str, default='/data/FiveK')
    parser.add_argument('--adaptor_checkpoint_dir', type=str, default='./checkpoint/RN50')
    parser.add_argument('--adaptor_checkpoint_name', type=str, default='pretrained.pth')
    parser.add_argument('--save_path', type=str, default='./test')
    parser.add_argument('--clip_model', type=str, default='RN50')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--intensity', type=float, default=1)
    parser.add_argument('--prompt', type=str, required=True)
    
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args = parser.parse_args()
    # DataLoader
    valid_dataset = SingleImageDataset(args.dataset_dir, '/data/FiveK/test.txt', aug=False)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True)
    
    CLIPloss = CLIPLoss(device, clip_model = 'RN50')
    
    model = AiLUT(args).to(device)
    model.load_state_dict(torch.load(os.path.join(args.backbone_checkpoint_dir, args.backbone_checkpoint_name))['state_dict'])
    model.requires_grad_(False)
    model.eval()
    adaptation_module = AdaptationModule(args, CLIPloss.model.text_projection.shape[1], model.backbone.out_channels).to(device)
    adaptation_module.load_state_dict(torch.load(os.path.join(args.adaptor_checkpoint_dir, args.adaptor_checkpoint_name)))
    adaptation_module.requires_grad_(False)
    adaptation_module.eval()
    
    
    save_path = os.path.join(args.save_path, args.prompt)
    target_text = args.prompt + ' photo.'
    text_direction_features = CLIPloss.compute_text_direction('Normal photo.', target_text)
    for i, (lq,  file_name) in enumerate(valid_loader):
        lq = lq.to(device)
        weights_deltas = adaptation_module(text_direction_features, args.intensity)
        out, _, _ = model(lq, weights_deltas = weights_deltas)
        file_name = os.path.splitext(file_name[0])[0]
                
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_image(out, os.path.join(save_path, f'{file_name}.png'))
        
if __name__ == '__main__':
    main()
