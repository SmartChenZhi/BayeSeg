import os
from monai.transforms.compose import Compose
from monai.transforms.io.dictionary import LoadImaged
from monai.transforms import (
    Orientationd,
    CenterSpatialCropd,
    Resized,
    NormalizeIntensityd,
    Spacingd,
    Transposed,
)
from data.transform import Mask2To1d, FilterOutBackgroundSliced, ClipHistogram
from monai.data import DataLoader, Dataset
import numpy as np
import cv2

# 输入文件夹路径
input_folder = "Processed_data_nii/test"
# 输出文件夹路径
output_train_folder = "/data4/smartchen/code/pytorch-CycleGAN-and-pix2pix/datasets/prostate/testB"
output_labels_folder = "A/val_lbs"

# 创建输出文件夹
os.makedirs(output_train_folder, exist_ok=True)
os.makedirs(output_labels_folder, exist_ok=True)

# 定义预处理流水线
preprocess = Compose(
    [
        LoadImaged(
            keys=["image", "label", "ori_image"],
            image_only=False,
            ensure_channel_first=True,
        ),
        FilterOutBackgroundSliced(
            keys=["image", "label", "ori_image"], source_key="label"
        ),
        Spacingd(
            keys="image", pixdim=(0.36458, 0.36458, -1), mode=("bilinear")
        ),
        ClipHistogram(keys="image", percentile=0.995),
        Orientationd(
            keys=["image", "label", "ori_image"], axcodes="PLS"
        ),  # orientation after spacing
        Mask2To1d(keys="label"),
        CenterSpatialCropd(keys="image", roi_size=[384, 384, -1]),
        Resized(keys="image", spatial_size=[192, 192, -1], mode=("trilinear")),
        Transposed(keys="image", indices=[3, 0, 1, 2]),
        NormalizeIntensityd(keys=["image", "ori_image"], channel_wise=True),
    ]
)

# 准备数据集
image_files = []
for file_name in os.listdir(input_folder):
    if file_name.endswith(".nii.gz") and not file_name.endswith("_segmentation.nii.gz"):
        image_path = os.path.join(input_folder, file_name)
        segmentation_path = image_path.replace(".nii.gz", "_segmentation.nii.gz")
        if os.path.exists(segmentation_path):
            image_files.append({"image": image_path, "label": segmentation_path, "ori_image": image_path})

dataset = Dataset(data=image_files, transform=preprocess)
data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

# 开始处理数据
for idx, batch in enumerate(data_loader):
    image = batch["image"][0].numpy()  # 获取图像
    label = batch["label"][0].numpy()  # 获取分割
    # print(image.shape)
    # print(label.shape)

    # 遍历每一层切片
    for i in range(image.shape[0]):  # 假设经过预处理后第 0 维是切片维度
        slice_image = image[i].squeeze(0)
        slice_label = label[:,:,:,i].squeeze(0)
        # print(slice_image.shape)
        # print(slice_label.shape)
        # breakpoint()

        # 转换为 uint8
        slice_image = ((slice_image - np.min(slice_image)) / (np.max(slice_image) - np.min(slice_image)) * 255).astype(np.uint8)
        slice_label = (slice_label / np.max(slice_label) * 255).astype(np.uint8) if np.max(slice_label) > 0 else slice_label.astype(np.uint8)

        # 保存切片为 JPG 文件
        slice_image_path = os.path.join(output_train_folder, f"case_{idx:02d}_slice_{i:02d}.jpg")
        #slice_label_path = os.path.join(output_labels_folder, f"case_{idx:02d}_slice_{i:02d}.jpg")

        cv2.imwrite(slice_image_path, slice_image)
        #cv2.imwrite(slice_label_path, slice_label)

    print(f"Processed case {idx + 1}/{len(image_files)}")

print("All files processed successfully!")