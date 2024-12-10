from models.networks import define_G
import torch
from PIL import Image
from torchvision import transforms

model = define_G(input_nc=1,output_nc=1,ngf=64,netG="resnet_9blocks",gpu_ids=[],norm="instance")
# 如果权重包含 "module." 前缀
state_dict = torch.load("runs/200_net_G_B.pth")
model.load_state_dict(state_dict)
# 切换到评估模式
model.eval()
# 读取 PNG 文件
image = Image.open("runs/case_00_slice_00_real_A.png").convert("L")  # 使用 PIL 加载图片
# 转换为张量
transform = transforms.ToTensor()
tensor = transform(image)
tensor = tensor.unsqueeze(0)

output = model(tensor)
output = output.squeeze(0)
# 转换为 PIL 图像
to_pil = transforms.ToPILImage()
image = to_pil(output)

# 保存为 PNG 文件
image.save("runs/grayscale_image.png")
