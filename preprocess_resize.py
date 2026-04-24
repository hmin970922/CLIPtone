"""
학습 전 이미지를 지정한 크기로 미리 리사이즈하여 저장하는 스크립트.
DataLoader에서 매번 리사이즈하는 비용을 제거해 학습 속도를 높입니다.

사용법:
    python preprocess_resize.py --src_dir ./data/FiveK/input/JPG/480p \
                                 --dst_dir ./data/FiveK/input/JPG/256p \
                                 --size 256
"""

import argparse
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image
from tqdm import tqdm


def resize_image(src_path: Path, dst_path: Path, size: int) -> str:
    try:
        img = Image.open(src_path).convert('RGB')
        # 짧은 쪽을 size에 맞추고 비율 유지 후 center crop
        w, h = img.size
        scale = size / min(w, h)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        img = img.resize((new_w, new_h), Image.BICUBIC)

        left = (new_w - size) // 2
        top  = (new_h - size) // 2
        img = img.crop((left, top, left + size, top + size))

        dst_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(dst_path, quality=95)
        return f'OK: {src_path.name}'
    except Exception as e:
        return f'ERR: {src_path.name} — {e}'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir', type=str, required=True,
                        help='원본 이미지 디렉토리 (예: ./data/FiveK/input/JPG/480p)')
    parser.add_argument('--dst_dir', type=str, required=True,
                        help='저장할 디렉토리 (예: ./data/FiveK/input/JPG/256p)')
    parser.add_argument('--size', type=int, default=256,
                        help='리사이즈할 정사각형 크기 (default: 256)')
    parser.add_argument('--ext', type=str, default='.jpg',
                        help='처리할 이미지 확장자 (default: .jpg)')
    parser.add_argument('--workers', type=int, default=os.cpu_count(),
                        help='병렬 워커 수 (default: CPU 코어 수)')
    parser.add_argument('--skip_existing', action='store_true',
                        help='이미 존재하는 파일은 건너뜀')
    args = parser.parse_args()

    src_dir = Path(args.src_dir)
    dst_dir = Path(args.dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    src_paths = sorted(src_dir.glob(f'*{args.ext}'))
    if not src_paths:
        print(f'[오류] {src_dir} 에서 {args.ext} 파일을 찾을 수 없습니다.')
        return

    tasks = []
    for src_path in src_paths:
        dst_path = dst_dir / src_path.name
        if args.skip_existing and dst_path.exists():
            continue
        tasks.append((src_path, dst_path))

    print(f'총 {len(src_paths)}개 중 {len(tasks)}개 처리 예정 → {dst_dir} (size={args.size})')

    errors = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(resize_image, s, d, args.size): s for s, d in tasks}
        for future in tqdm(as_completed(futures), total=len(futures), desc='Resizing'):
            result = future.result()
            if result.startswith('ERR'):
                errors.append(result)

    print(f'\n완료: {len(tasks) - len(errors)}개 성공, {len(errors)}개 실패')
    for e in errors:
        print(f'  {e}')


if __name__ == '__main__':
    main()
