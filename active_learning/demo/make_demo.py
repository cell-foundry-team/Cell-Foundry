"""Create 10 FAKE candidates (noise + circles) in demo_cycle/ so the review page can be tested with no real data."""
import csv, random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter
random.seed(7)
out = Path(__file__).parent / "demo_cycle"; (out / "thumbnails_layers").mkdir(parents=True, exist_ok=True); (out / "thumbnails_for_expert").mkdir(parents=True, exist_ok=True)
rows = []
for k in range(1, 11):
    base = Image.effect_noise((512, 512), 25).convert("RGB").filter(ImageFilter.GaussianBlur(3))
    base = Image.blend(base, Image.new("RGB", (512, 512), (150, 150, 150)), 0.55)
    ImageDraw.Draw(base).text((10, 10), "FAKE TEST IMAGE - NOT REAL DATA", fill=(255, 255, 255))
    gt = Image.new("RGBA", (512, 512), (0, 0, 0, 0)); dg = ImageDraw.Draw(gt)
    for _ in range(6):
        x, y, r = random.randint(60, 452), random.randint(60, 452), random.randint(36, 68)
        dg.ellipse([x - r, y - r, x + r, y + r], outline=(0, 255, 0, 255), width=2)
    cand = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
    x, y, r = random.randint(140, 372), random.randint(140, 372), random.randint(40, 60)
    ImageDraw.Draw(cand).ellipse([x - r, y - r, x + r, y + r], outline=(255, 215, 0, 255), width=3)
    cid = f"C{k:03d}"
    base.save(out / "thumbnails_layers" / f"{cid}_base.png"); gt.save(out / "thumbnails_layers" / f"{cid}_gt.png"); cand.save(out / "thumbnails_layers" / f"{cid}_cand.png")
    comp = base.convert("RGBA"); comp.alpha_composite(gt); comp.alpha_composite(cand); comp.convert("RGB").save(out / "thumbnails_for_expert" / f"{cid}.png")
    import json, math
    json.dump([[round(x + r * math.cos(2 * math.pi * j / 64), 1), round(y + r * math.sin(2 * math.pi * j / 64), 1)] for j in range(64)],
              open(out / "thumbnails_layers" / f"{cid}_poly.json", "w"))
    rows.append(dict(candidate_id=cid, image=f"DEMO_image_{(k % 4) + 1}", thumbnail=f"{cid}.png", crop_x=100 * k, crop_y=50 * k))
with open(out / "review.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["candidate_id", "image", "thumbnail", "crop_x", "crop_y"]); w.writeheader(); w.writerows(rows)
print("10 fake candidates written to", out)
