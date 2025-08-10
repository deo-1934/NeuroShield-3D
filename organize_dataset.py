import os
import shutil

# مسیر اصلی دیتاست
dataset_dir = r"D:\NeuroShield-3D\data\raw\Task01_BrainTumour"
train_dir = os.path.join(dataset_dir, "imagesTr")
label_dir = os.path.join(dataset_dir, "labelsTr")

# اگه پوشه‌ها وجود نداشت، بسازشون
os.makedirs(train_dir, exist_ok=True)
os.makedirs(label_dir, exist_ok=True)

# پیمایش همه پوشه‌ها و فایل‌ها
for root, _, files in os.walk(dataset_dir):
    for file in files:
        if file.endswith(".nii") or file.endswith(".nii.gz"):
            src_path = os.path.join(root, file)
            if "_seg" in file.lower():  # فایل لیبل
                shutil.move(src_path, os.path.join(label_dir, file))
            else:  # فایل MRI
                shutil.move(src_path, os.path.join(train_dir, file))

print("✅ تمام فایل‌ها مرتب شدند!")
