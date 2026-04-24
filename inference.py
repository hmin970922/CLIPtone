"""
단일 이미지 또는 폴더에 대해 스타일 변환을 수행하는 inference 스크립트.

사용법:
    # 단일 이미지
    python inference.py --input ./photo.jpg --prompt "Warm"

    # 폴더 전체
    python inference.py --input ./photos/ --prompt "Cold" --save_path ./output

    # intensity 조절 (0.0 ~ 2.0 권장)
    python inference.py --input ./photo.jpg --prompt "High contrast" --intensity 1.5
"""

import argparse
import os
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm

from ailutmodel import AiLUT
from adaptation import AdaptationModule
from criteria import CLIPLoss

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}


def load_image(path: Path) -> torch.Tensor:
    img = Image.open(path).convert('RGB')
    return transforms.ToTensor()(img).unsqueeze(0)  # (1, 3, H, W)


def collect_images(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(p for p in input_path.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input', type=str, required=True,
                        help='입력 이미지 경로 또는 폴더')
    parser.add_argument('--prompt', type=str, required=True,
                        help='스타일 프롬프트 (예: "Warm", "Cold", "High contrast")')
    parser.add_argument('--save_path', type=str, default='./output',
                        help='결과 저장 폴더 (default: ./output)')

    parser.add_argument('--backbone', type=str, default='tpami')
    parser.add_argument('--backbone_checkpoint_dir', type=str, default='./checkpoint/base_network')
    parser.add_argument('--backbone_checkpoint_name', type=str, default='AiLUT-FiveK-sRGB.pth')
    parser.add_argument('--n_ranks', type=int, default=3)
    parser.add_argument('--n_vertices', type=int, default=33)
    parser.add_argument('--n_colors', type=int, default=3)
    parser.add_argument('--en_adaint', type=bool, default=True)
    parser.add_argument('--en_adaint_share', type=bool, default=False)
    parser.add_argument('--pretrained', type=bool, default=False)

    parser.add_argument('--adaptor_checkpoint_dir', type=str,
                        default='./checkpoint/text_adaptor/MobileCLIP2-S0')
    parser.add_argument('--adaptor_checkpoint_name', type=str, default='pretrained.pth')
    parser.add_argument('--clip_model', type=str, default='MobileCLIP2-S0')
    parser.add_argument('--clip_pretrained', type=str, default='dfndr2b')
    parser.add_argument('--intensity', type=float, default=1.0,
                        help='스타일 적용 강도 (default: 1.0, 권장 범위: 0.0 ~ 2.0)')

    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    args = parser.parse_args()

    # 모델 로드
    print(f'[1/3] 모델 로딩 중...')
    clip_loss = CLIPLoss(device, clip_model=args.clip_model, clip_pretrained=args.clip_pretrained)

    model = AiLUT(args).to(device)
    model.load_state_dict(
        torch.load(os.path.join(args.backbone_checkpoint_dir, args.backbone_checkpoint_name),
                   map_location=device)['state_dict']
    )
    model.requires_grad_(False)
    model.eval()

    adaptation_module = AdaptationModule(args, clip_loss.text_dim, model.backbone.out_channels).to(device)
    adaptation_module.load_state_dict(
        torch.load(os.path.join(args.adaptor_checkpoint_dir, args.adaptor_checkpoint_name),
                   map_location=device)
    )
    adaptation_module.requires_grad_(False)
    adaptation_module.eval()

    # 텍스트 방향 벡터 계산
    print(f'[2/3] 텍스트 방향 계산 중: "{args.prompt} photo."')
    with torch.no_grad():
        text_direction = clip_loss.compute_text_direction('Normal photo.', args.prompt + ' photo.')
        weights_deltas = adaptation_module(text_direction, args.intensity)

    # 이미지 추론
    input_path = Path(args.input)
    images = collect_images(input_path)
    if not images:
        print(f'[오류] {input_path} 에서 이미지를 찾을 수 없습니다.')
        return

    save_dir = Path(args.save_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f'[3/3] {len(images)}개 이미지 처리 중 → {save_dir}')
    with torch.no_grad():
        for img_path in tqdm(images, desc='Inference'):
            lq = load_image(img_path).to(device)
            out, _, _ = model(lq, weights_deltas=weights_deltas)
            out_path = save_dir / (img_path.stem + '.png')
            save_image(out, out_path)

    print(f'완료: {save_dir}')


if __name__ == '__main__':
    main()
