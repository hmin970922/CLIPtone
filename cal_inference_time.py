import torch
import time
import argparse
import os
import numpy as np
from dataset.dataset import SingleDataset, TensorDataset
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from ailutmodel import AiLUT
from adaptation import AdaptationModule
from criteria import CLIPLoss

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--n_ranks', type=int, default=3)
    parser.add_argument('--n_vertices', type=int, default=33)
    parser.add_argument('--en_adaint', type=bool, default=True)
    parser.add_argument('--en_adaint_share', type=bool, default=False)
    parser.add_argument('--backbone', type=str, default='tpami')
    parser.add_argument('--pretrained', type=bool, default=False)
    parser.add_argument('--n_colors', type=int, default=3)
    parser.add_argument('--sparse_factor', type=float, default=0.0001)
    parser.add_argument('--smooth_factor', type=float, default=0.5)
    parser.add_argument('--monotonicity_factor', type=float, default=10.0)
    parser.add_argument('--dataset', type=str, default='/data/FiveK/')
    parser.add_argument('--save_path', type=str, default='/log/transform/result/finetuned_RN50_test')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint/pretrained')
    parser.add_argument('--checkpoint_name', type=str, default='latest.pth')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_epoch', type=int, default=400)
    parser.add_argument('--print_freq', type=int, default=100)
    parser.add_argument('--valid_epoch_freq', type=int, default=1)
    parser.add_argument('--tensor_file_path', type=str, default='./csv/direction_features.pth')
    parser.add_argument('--text', type=str, default='Neon light photo.')
    parser.add_argument('--intensity', type=float, default=1)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args = parser.parse_args()
    # DataLoader
    valid_dataset = SingleDataset(args.dataset, '/data/FiveK/test.txt', aug=False)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True)
    
    model = AiLUT(args).to(device)
    model.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, args.checkpoint_name))['state_dict'])
    model.requires_grad_(False)
    model.eval()
    adaptation_module = AdaptationModule(args, model.backbone.out_channels).to(device)
    adaptation_module.requires_grad_(False)
    adaptation_module.eval()

    CLIPloss = CLIPLoss(device)
    save_path = os.path.join(args.save_path, ' '.join(args.text.split(" ")[:-1]))
    
    dummy_input = torch.randn(1,3,512,512, dtype=torch.float).cuda()
    dummy_direction = torch.randn(1, 1024, dtype=torch.float).cuda()
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 300
    timings=np.zeros((repetitions,1))
    
    num_param = sum(p.numel() for p in CLIPloss.model.parameters())
    print(num_param)
    num_param = sum(p.numel() for p in CLIPloss.model.visual.parameters())
    print(num_param)
    num_param = sum(p.numel() for p in CLIPloss.model.token_embedding.parameters())
    print(num_param)
    num_param = sum(p.numel() for p in CLIPloss.model.positional_embedding)
    print(num_param)
    num_param = sum(p.numel() for p in CLIPloss.model.transformer.parameters())
    print(num_param)
    num_param = sum(p.numel() for p in CLIPloss.model.ln_final.parameters())
    print(num_param)
    num_param = sum(p.numel() for p in model.parameters())
    print(num_param)
    num_param = sum(p.numel() for p in adaptation_module.parameters())
    print(num_param)
    
    
    for _ in range(10):
        dummy_weighs_delta = adaptation_module(dummy_direction)
        _ = model(dummy_input, weights_deltas = dummy_weighs_delta)
        
    dummy_tensor1 = CLIPloss.get_text_features('Normal photo.')
    # MEASURE PERFORMANCE
    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            dummy_tensor2 = CLIPloss.get_text_features('Hi')
            dummy_direction = CLIPloss.compute_text_direction_feature(dummy_tensor1, dummy_tensor2)
            dummy_weighs_delta = adaptation_module(dummy_direction)
            _ = model(dummy_input, weights_deltas = dummy_weighs_delta)
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time
    mean_syn = np.sum(timings) / repetitions
    
    print(mean_syn)
    # for i, (lq,  file_name) in enumerate(valid_loader):
    #     iter_start_time = time.time()
        
    #     lq = lq.to(device)
    #     text_direction_features = CLIPloss.compute_text_direction('Normal photo.', args.text)
    #     weights_deltas = adaptation_module(text_direction_features, args.intensity)
    #     out, _, _ = model(lq, weights_deltas = weights_deltas)
    #     file_name = os.path.splitext(file_name[0])[0]
                
                
    #     if not os.path.exists(save_path):
    #         os.makedirs(save_path)
    #     save_image(out, os.path.join(save_path, f'{file_name}.png'))


if __name__ == '__main__':
    main()
