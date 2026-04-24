import torch
import argparse
import os
from dataset.dataset import SingleImageDataset
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
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
    parser.add_argument('--dataset_dir', type=str, default='./data/FiveK')
    parser.add_argument('--img_subdir', type=str, default='input/JPG/480p')
    parser.add_argument('--test_list', type=str, default=None,
                        help='테스트 파일 목록 (기본값: dataset_dir/test.txt)')
    parser.add_argument('--adaptor_checkpoint_dir', type=str, default='./checkpoint/text_adaptor/MobileCLIP2-S0')
    parser.add_argument('--adaptor_checkpoint_name', type=str, default='pretrained.pth')
    parser.add_argument('--save_path', type=str, default='./test')
    parser.add_argument('--clip_model', type=str, default='MobileCLIP2-S0')
    parser.add_argument('--clip_pretrained', type=str, default='dfndr2b')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--intensity', type=float, default=1.0)
    parser.add_argument('--prompt', type=str, required=True,
                        help='스타일 프롬프트 (예: "Warm", "Cold", "High contrast")')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args = parser.parse_args()

    test_list = args.test_list or os.path.join(args.dataset_dir, 'test.txt')
    valid_dataset = SingleImageDataset(
        args.dataset_dir, test_list, aug=False, img_subdir=args.img_subdir
    )
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)

    CLIPloss = CLIPLoss(device, clip_model=args.clip_model, clip_pretrained=args.clip_pretrained)

    model = AiLUT(args).to(device)
    model.load_state_dict(
        torch.load(os.path.join(args.backbone_checkpoint_dir, args.backbone_checkpoint_name))['state_dict']
    )
    model.requires_grad_(False)
    model.eval()

    adaptation_module = AdaptationModule(args, CLIPloss.text_dim, model.backbone.out_channels).to(device)
    adaptation_module.load_state_dict(
        torch.load(os.path.join(args.adaptor_checkpoint_dir, args.adaptor_checkpoint_name))
    )
    adaptation_module.requires_grad_(False)
    adaptation_module.eval()

    save_path = os.path.join(args.save_path, args.prompt.replace(' ', '_'))
    os.makedirs(save_path, exist_ok=True)

    target_text = args.prompt + ' photo.'
    with torch.no_grad():
        text_direction = CLIPloss.compute_text_direction('Normal photo.', target_text)
        weights_deltas = adaptation_module(text_direction, args.intensity)

        for lq, file_name in tqdm(valid_loader, desc='Testing'):
            lq = lq.to(device)
            out, _, _ = model(lq, weights_deltas=weights_deltas)
            stem = os.path.splitext(file_name[0])[0]
            save_image(out, os.path.join(save_path, f'{stem}.png'))


if __name__ == '__main__':
    main()
