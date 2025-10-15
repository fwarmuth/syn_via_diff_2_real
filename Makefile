.PHONY: sync gen lora fid qa test

sync:
uv sync

gen:
uv run python -m syn_via_diff.pipeline --config configs/default.yaml

lora:
uv run python scripts/train_lora.py --config configs/lora.yaml

fid:
uv run python scripts/compute_fid.py data/realistic/images data/real_ref/images

qa:
uv run python - <<'PY'
from pathlib import Path
from PIL import Image
from syn_via_diff.viz import overlay_mask

out_dir = Path('data/realistic/images')
mask_dir = Path('data/realistic/masks')
qa_dir = Path('artifacts/qa')
qa_dir.mkdir(parents=True, exist_ok=True)
for image_path in out_dir.rglob('*.png'):
    rel = image_path.relative_to(out_dir)
    mask_path = mask_dir / rel
    if not mask_path.exists():
        continue
    with Image.open(image_path) as image, Image.open(mask_path) as mask:
        panel = overlay_mask(image, mask)
        out_path = qa_dir / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        panel.save(out_path)
PY

test:
uv run pytest
