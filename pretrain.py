import torch
import time
import argparse
import os
from dataset.dataset import PairedDataset
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from model import AiLUT

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
    parser.add_argument('--smooth_factor', type=float, default=0)
    parser.add_argument('--monotonicity_factor', type=float, default=10.0)
    parser.add_argument('--dataset', type=str, default='/data/FiveK/')
    parser.add_argument('--save_path', type=str, default='./result/pretrained')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_epoch', type=int, default=400)
    parser.add_argument('--print_freq', type=int, default=100)
    parser.add_argument('--valid_epoch_freq', type=int, default=20)
    
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # DataLoader
    train_dataset = PairedDataset(args.dataset, '../data/FiveK/train.txt', aug=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    valid_dataset = PairedDataset(args.dataset, '../data/FiveK/test.txt', aug=False)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True)
    
    
    model = AiLUT(args).to(device)
    optimizer = torch.optim.Adam([{'params': model.backbone.parameters()},
                                  {'params': model.lut_generator.parameters()},
                                  {'params': model.adaint.parameters(), 'lr': 1e-5}],
                                 lr=1e-4, weight_decay=0, betas=(0.9, 0.999), eps=1e-8)
    MSEloss = torch.nn.MSELoss()

    total_steps = 0
    sum_l1_loss = 0.0
    sum_sparse_loss = 0.0
    sum_smooth_loss = 0.0
    sum_monotonicity_loss = 0.0
    for epoch in range(1, args.num_epoch):
        model.requires_grad_(False)
        model.train()
        epoch_start_time = time.time()
        
        for i, (lq, gt, file_name) in enumerate(train_loader):
            iter_start_time = time.time()
            total_steps += args.batch_size
            
            lq = lq.to(device)
            gt = gt.to(device)
            out, weights, vertices = model(lq, test_mode=False)
            l1_loss = MSEloss(out, gt)
            
            if args.sparse_factor > 0:
                sparse_loss = args.sparse_factor * torch.mean(weights.pow(2))
            else:
                sparse_loss = 0.0
                
            reg_smooth, reg_monotonicity = model.lut_generator.regularizations(vertices, interval_adaptive=False)
            
            if args.smooth_factor > 0:
                smooth_loss = args.smooth_factor * reg_smooth
            else:
                smooth_loss = 0.0
                
            if args.monotonicity_factor > 0:
                monotonicity_loss = args.monotonicity_factor * reg_monotonicity
            else:
                monotonicity_loss = 0.0
                
            total_loss = l1_loss + sparse_loss + smooth_loss + monotonicity_loss
            
            sum_l1_loss += l1_loss
            sum_sparse_loss += sparse_loss
            sum_smooth_loss += smooth_loss
            sum_monotonicity_loss += monotonicity_loss
                
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            if total_steps % args.print_freq == 0:
                sum_loss = sum_l1_loss + sum_sparse_loss + sum_smooth_loss + sum_monotonicity_loss
                print(f'iter = {total_steps}, sum_loss = {sum_loss / args.print_freq:.6f}, l1 = {sum_l1_loss / args.print_freq:.6f}, sparse = {sum_sparse_loss / args.print_freq:.6f}, \
smooth = {sum_smooth_loss / args.print_freq:.6f}, monotonicity = {sum_monotonicity_loss / args.print_freq:.6f}')
                
                sum_l1_loss = 0.0
                sum_sparse_loss = 0.0
                sum_smooth_loss = 0.0
                sum_monotonicity_loss = 0.0
        
        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, args.num_epoch, time.time() - epoch_start_time))
        
        if epoch % args.valid_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
            torch.save(model.state_dict(), f'./checkpoint/pretrained/{epoch}.pth')
            
            model.requires_grad_(False)
            model.eval()
            
            for i, (lq, gt, file_name) in enumerate(valid_loader):
                iter_start_time = time.time()
                
                lq = lq.to(device)
                out, _, _ = model(lq, test_mode=True)
                file_name = os.path.splitext(file_name[0])[0]
                save_path = os.path.join(args.save_path, file_name)
                
                
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                save_image(out, os.path.join(save_path, f'{file_name}-{total_steps:06d}.png'))


if __name__ == '__main__':
    main()
