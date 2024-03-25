import torch
import time
import argparse
import os
from dataset.dataset import SingleBatchDataset, SingleDataset, TensorDataset
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from ailutmodel import AiLUT
from adaptation_extend import AdaptationModule
from criteria import CLIPLoss #, ColorLoss

import torch.nn as nn
import random
import wandb
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
    parser.add_argument('--pretrained_dir', type=str, default='./checkpoint/pretrained')
    parser.add_argument('--pretrained_name', type=str, default='latest.pth')
    parser.add_argument('--save_path', type=str, default='/log/CLIPtone/4layer/0.0001_0.5')
    parser.add_argument('--clip_model', type=str, default='RN50')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epoch', type=int, default=400)
    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument('--valid_epoch_freq', type=int, default=40)
    parser.add_argument('--tensor_file_path', type=str, default='./csv/target_direction_RN50.pth')
    # parser.add_argument('--tensor_file_path', type=str, default='./csv/text_features_RN50_new.pth')
    
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args = parser.parse_args()
    # DataLoader
    train_dataset = SingleBatchDataset(args.dataset, os.path.join(args.dataset,'train.txt'), aug=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    valid_dataset = SingleDataset(args.dataset, os.path.join(args.dataset,'test.txt'), aug=False)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=True)
    
    train_text_dataset = TensorDataset(args.tensor_file_path)
    train_text_loader = DataLoader(train_text_dataset, batch_size=1, shuffle=True)
    
    MSEloss = torch.nn.MSELoss()
    CLIPloss = CLIPLoss(device, clip_model = args.clip_model)
    # Colorloss = ColorLoss(device)
    
    model = AiLUT(args).to(device)
    model.load_state_dict(torch.load(os.path.join(args.pretrained_dir, args.pretrained_name))['state_dict'])
    model.requires_grad_(False)
    
    adaptation_module = AdaptationModule(args, CLIPloss.model.text_projection.shape[1], model.backbone.out_channels).to(device)
    
    
    optimizer = torch.optim.Adam(adaptation_module.parameters(), lr=1e-3, weight_decay=0, betas=(0.9, 0.999), eps=1e-8)
    
    args.save_path = os.path.join(args.save_path, args.clip_model)
    if not os.path.exists(os.path.join(args.save_path, 'checkpoint')):
        os.makedirs(os.path.join(args.save_path, 'checkpoint'))
        
    train_text_prompts = ['Aged photo.', 'Cinematic photo.', 'Grayscale photo.', 'Natural photo.', 'Vivid photo.', 'Matte photo.', 'Bright photo.']
     
    test_text_prompts = ['Warm photo.', 'Cold photo.', 'Bright photo.', 'Dark photo.', 'Monotone photo.', 'Colorful photo.', 'Vivid photo.', 'High contrast photo.']
    
  
    wandb.init(project="CLIPtone", name="6layer_0.0001_0.5", entity="hyeongmin")
    
    total_steps = 0
    sum_l1_loss, sum_clip_loss, sum_sparse_loss, sum_smooth_loss, sum_monotonicity_loss, sum_total_loss = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    
    count = 0
    # sum_color_loss = 0.0
    
    for epoch in range(1, args.num_epoch + 1):
        model.train()
        epoch_start_time = time.time()
        adaptation_module.requires_grad_(True)
        adaptation_module.train()
        for i, ((lq, file_name), target_direction) in enumerate(zip(train_loader, train_text_loader)):
            iter_start_time = time.time()
            batch_size = lq.shape[0]
            total_steps += batch_size
            file_name = os.path.splitext(file_name[0])[0]
            lq = lq.to(device)
            target_direction = target_direction.to(device)
            weights_deltas = adaptation_module(target_direction)
            out, weights, vertices = model(lq, weights_deltas=weights_deltas)
            
            l1_loss = MSEloss(lq, out)
            clip_loss = CLIPloss(lq, out, target_direction)
            
            if args.sparse_factor > 0:
                sparse_loss = args.sparse_factor * torch.mean(weights.pow(2))
            else:
                sparse_loss = 0.0
            
            reg_smooth, reg_monotonicity = model.lut_generator.regularizations(vertices, interval_adaptive = True)
            if args.smooth_factor > 0:
                smooth_loss = args.smooth_factor * reg_smooth
            else:
                smooth_loss = 0.0
            
            if args.monotonicity_factor > 0:
                monotonicity_loss = args.monotonicity_factor * reg_monotonicity
            else:
                monotonicity_loss = 0.0
            
            total_loss = l1_loss + clip_loss + sparse_loss + smooth_loss + monotonicity_loss # + color_loss
            optimizer.zero_grad()
            total_loss.backward()
            # import pdb; pdb.set_trace()
            # norm = torch.nn.utils.clip_grad_norm_(adaptation_module.parameters(), 1)
            optimizer.step()
            
            count += 1
            sum_l1_loss += l1_loss.item()
            sum_clip_loss += clip_loss.item()
            sum_sparse_loss += sparse_loss.item()
            sum_smooth_loss += smooth_loss.item()
            sum_monotonicity_loss += monotonicity_loss.item()
            sum_total_loss += total_loss.item()
            
            # sum_color_loss += color_loss
            if count % args.print_freq == 0:
                print(f'iter = {total_steps}, total_loss = {total_loss.item():.6f}, l1 = {l1_loss.item():.6f}, clip = {clip_loss.item():.6f}, \
sparse = {sparse_loss.item():.6f}, smooth = {smooth_loss.item():.6f}, monotonicity = {monotonicity_loss.item():.6f}')
       
        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, args.num_epoch, time.time() - epoch_start_time))
        
        avg_l1_loss = sum_l1_loss / count
        avg_clip_loss = sum_clip_loss / count
        avg_sparse_loss = sum_sparse_loss / count
        avg_smooth_loss = sum_smooth_loss / count
        avg_monotonicity_loss = sum_monotonicity_loss / count
        avg_total_loss = sum_total_loss / count
        count = 0
        
        wandb.log({"epoch": epoch, "total": avg_total_loss, "l1": avg_l1_loss, "clip": avg_clip_loss, "sparse": avg_sparse_loss, "smooth": avg_smooth_loss, "monotonicity": avg_monotonicity_loss})
        sum_l1_loss, sum_clip_loss, sum_sparse_loss, sum_smooth_loss, sum_monotonicity_loss, sum_total_loss = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        
        if epoch % args.valid_epoch_freq == 0:
            model.eval()
            adaptation_module.requires_grad_(False)
            adaptation_module.eval()
            
            sum_clip_direction = 0.0
            for i, (lq,  file_name) in enumerate(valid_loader):
                iter_start_time = time.time()
                lq = lq.to(device)
                # image_feature = CLIPloss.get_image_features(lq)
                file_name = os.path.splitext(file_name[0])[0]
                # index = 0
                for index in range(0, len(test_text_prompts)):
                    target_direction = CLIPloss.compute_text_direction('Normal photo.', test_text_prompts[index])
                    weights_deltas = adaptation_module(target_direction)
                    # weights_deltas = adaptation_module(torch.cat((target_direction, image_feature)).flatten())
                    out, _, _ = model(lq, weights_deltas = weights_deltas)
                    sum_clip_direction += CLIPloss(lq, out, target_direction)
                    save_path = os.path.join(args.save_path, 'result', file_name)
                
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                        
                    save_image(out, os.path.join(save_path, f'{file_name}-{total_steps:06d}_{test_text_prompts[index]}.png'))
            print(f'valid loss = {sum_clip_direction / (len(valid_loader) * len(test_text_prompts))}')
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
            torch.save(adaptation_module.state_dict(), f'{args.save_path}/checkpoint/{epoch}_{sum_clip_direction / len(valid_loader)}.pth')


if __name__ == '__main__':
    main()
