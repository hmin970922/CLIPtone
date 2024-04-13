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
    parser.add_argument('--n_ranks', type=int, default=3, help='The number of basis LUTs.')
    parser.add_argument('--n_vertices', type=int, default=33, help='The number of sampling points along each lattice dimension.')
    parser.add_argument('--n_colors', type=int, default=3, help='The number of input color channels.')
    parser.add_argument('--content_factor', type=float, default=1.0, help='Loss weight for the content loss term.')
    parser.add_argument('--clip_factor', type=float, default=1.0, help='Loss weight for the CLIP loss term.')
    parser.add_argument('--weight_factor', type=float, default=0.0001, help='Loss weight for weight loss term.')
    parser.add_argument('--interval_factor', type=float, default=0.5, help='Loss weight for interval loss term.')
    parser.add_argument('--monotonicity_factor', type=float, default=0.0, help = 'Loss weight for the monotonicity regularization term.' )
    parser.add_argument('--dataset_dir', type=str, default='/data/FiveK/')
    parser.add_argument('--save_path', type=str, default='./')
    parser.add_argument('--clip_model', type=str, default='RN50')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_epoch', type=int, default=400)
    parser.add_argument('--print_iter_freq', type=int, default=100)
    parser.add_argument('--valid_epoch_freq', type=int, default=40)
    parser.add_argument('--direction_dataset_path', type=str, default='./csv/target_direction_RN50.pth')
    parser.add_argument('--wandb', type=bool, default=False, help='logging with wandb')
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args = parser.parse_args()
    
    if args.wandb:    
        import wandb
        wandb.init(project="CLIPtone", name="CLIPtone")
    
    # DataLoader
    train_dataset = SingleImageDataset(args.dataset_dir, os.path.join(args.dataset_dir,'train.txt'), aug=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    valid_dataset = SingleImageDataset(args.dataset_dir, os.path.join(args.dataset_dir,'test.txt'), aug=False)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True)
    
    train_text_dataset = DirectionDataset(args.direction_dataset_path)
    train_text_loader = DataLoader(train_text_dataset, batch_size=args.batch_size, shuffle=True)
    
    MSEloss = torch.nn.MSELoss()
    CLIPloss = CLIPLoss(device, clip_model = args.clip_model)
    
    model = AiLUT(args).to(device)
    model.load_state_dict(torch.load(os.path.join(args.backbone_checkpoint_dir, args.backbone_checkpoint_name))['state_dict'])
    model.requires_grad_(False)
    
    adaptation_module = AdaptationModule(args, CLIPloss.model.text_projection.shape[1], model.backbone.out_channels).to(device)
    optimizer = torch.optim.Adam(adaptation_module.parameters(), lr=1e-3, weight_decay=0, betas=(0.9, 0.999), eps=1e-8)
    
    save_checkpoint_path = os.path.join(args.save_path, 'checkpoint', 'text_adaptor', args.clip_model)
    save_result_path = os.path.join(args.save_path, 'result', args.clip_model)
    
    if not os.path.exists(save_checkpoint_path):
        os.makedirs(save_checkpoint_path)
     
    test_text_prompts = ['Warm photo.', 'Cold photo.', 'High contrast photo.']
    
    total_steps = 0
    sum_content_loss, sum_clip_loss, sum_weight_loss, sum_interval_loss, sum_monotonicity_loss, sum_total_loss = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    iteration_count = 0
    
    for epoch in range(1, args.num_epoch + 1):
        model.train()
        epoch_start_time = time.time()
        adaptation_module.requires_grad_(True)
        adaptation_module.train()
        for i, ((lq, file_name), target_direction) in enumerate(zip(train_loader, train_text_loader)):
            total_steps += args.batch_size
            file_name = os.path.splitext(file_name[0])[0]
            lq = lq.to(device)
            target_direction = target_direction.to(device)
            weights_deltas = adaptation_module(target_direction)
            out, weights, vertices = model(lq, weights_deltas=weights_deltas)
            
            if args.content_factor > 0:
                content_loss = args.content_factor * MSEloss(lq, out)
            else:
                content_loss = 0.0
            
            if args.clip_factor > 0:
                clip_loss = args.clip_factor * CLIPloss(lq, out, target_direction)
            else:
                clip_loss = 0.0
            
            reg_interval, reg_monotonicity = model.lut_generator.regularizations(vertices, interval_adaptive = True)
            
            if args.weight_factor > 0:
                weight_loss = args.weight_factor * torch.mean(weights.pow(2))
            else:
                weight_loss = 0.0
            
            if args.interval_factor > 0:
                interval_loss = args.interval_factor * reg_interval
            else:
                interval_loss = 0.0
                
            if args.monotonicity_factor > 0:
                monotonicity_loss = args.monotonicity_factor * reg_monotonicity
            else:
                monotonicity_loss = 0.0
                
                
            total_loss = content_loss + clip_loss + weight_loss + interval_loss + monotonicity_loss
        
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            
            iteration_count += 1
            sum_content_loss += content_loss.item()
            sum_clip_loss += clip_loss.item()
            sum_weight_loss += weight_loss.item()
            sum_interval_loss += interval_loss.item()
            sum_monotonicity_loss += monotonicity_loss
            sum_total_loss += total_loss.item()
            
            if iteration_count % args.print_iter_freq == 0:
                print(f'iter = {total_steps}, total_loss = {total_loss.item():.6f}, content = {content_loss.item():.6f}, clip = {clip_loss.item():.6f}, \
weight = {weight_loss.item():.6f}, interval = {interval_loss.item():.6f}, monotonicity = {monotonicity_loss:.6f}')
                
        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, args.num_epoch, time.time() - epoch_start_time))
        
        avg_content_loss = sum_content_loss / iteration_count
        avg_clip_loss = sum_clip_loss / iteration_count
        avg_weight_loss = sum_weight_loss / iteration_count
        avg_interval_loss = sum_interval_loss / iteration_count
        avg_monotonicity_loss = sum_monotonicity_loss / iteration_count
        avg_total_loss = sum_total_loss / iteration_count
        iteration_count = 0
        
        if args.wandb:
            wandb.log({"epoch": epoch, "total": avg_total_loss, "content": avg_content_loss, "clip": avg_clip_loss, "weight": avg_weight_loss, "interval": avg_interval_loss, "monotonicity": avg_monotonicity_loss})
            
        sum_content_loss, sum_clip_loss, sum_weight_loss, sum_interval_loss, sum_monotonicity_loss, sum_total_loss = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        
        if epoch % args.valid_epoch_freq == 0:
            model.eval()
            adaptation_module.requires_grad_(False)
            adaptation_module.eval()
            
            for i, (lq,  file_name) in enumerate(valid_loader):
                lq = lq.to(device)
                file_name = os.path.splitext(file_name[0])[0]
                for index in range(0, len(test_text_prompts)):
                    target_direction = CLIPloss.compute_text_direction('Normal photo.', test_text_prompts[index])
                    weights_deltas = adaptation_module(target_direction)
                    out, _, _ = model(lq, weights_deltas = weights_deltas)
                    save_path = os.path.join(save_result_path, file_name)
                
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                        
                    save_image(out, os.path.join(save_path, f'{file_name}-{total_steps:06d}_{test_text_prompts[index]}.png'))
            
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
            torch.save(adaptation_module.state_dict(), f'{save_checkpoint_path}/{epoch}.pth')



if __name__ == '__main__':
    main()
