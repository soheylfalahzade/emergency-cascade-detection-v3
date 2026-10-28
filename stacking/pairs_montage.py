from pathlib import Path
from PIL import Image
D = Path.home() / "Portfolio/Smart_Traffic_System/datasets/folds/fold_0"
P = [("uadetrac_yolo_018634.jpg","uadetrac_yolo_017936.jpg"),("uadetrac_yolo_018431.jpg","uadetrac_yolo_017800.jpg"),
     ("emergency_vehicles_raw_000578.jpg","emergency_vehicles_raw_000333.jpg"),("emergency_vehicles_raw_000284.jpg","emergency_vehicles_raw_000206.jpg"),
     ("kitti_yolo_010286.png","kitti_yolo_015566.png"),("uadetrac_yolo_018599.jpg","uadetrac_yolo_017882.jpg")]
W, H = 480, 300
sheet = Image.new("RGB", (2 * W, len(P) * H), "white")
for r, (v, t) in enumerate(P):
    for c, p in enumerate((D / "val/images" / v, D / "train/images" / t)):
        sheet.paste(Image.open(p).convert("RGB").resize((W, H)), (c * W, r * H))
sheet.save(Path.home() / "leak_pairs.png"); print("saved ~/leak_pairs.png (ستون چپ=val، راست=train)")
