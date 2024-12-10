import torch
import torch.nn as nn


# VQ-VAE 中的向量量化层
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        # 创建代码本，形状为 [num_embeddings, embedding_dim]
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings)

    def forward(self, inputs):
        # 1. 将输入展平为 [batch, height, width, embedding_dim]
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self.embedding_dim)

        # 2. 计算每个输入到所有嵌入的距离并找到最近的嵌入
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True) +
                     torch.sum(self.embedding.weight ** 2, dim=1) -
                     2 * torch.matmul(flat_input, self.embedding.weight.t()))

        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        quantized = self.embedding(encoding_indices).view(input_shape)

        # 3. 损失计算
        e_latent_loss = torch.mean((quantized.detach() - inputs) ** 2)
        q_latent_loss = torch.mean((quantized - inputs.detach()) ** 2)
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # 4. 将量化后的向量的梯度替换为输入的梯度
        quantized = inputs + (quantized - inputs).detach()
        return quantized.permute(0, 3, 1, 2).contiguous(), loss

# VQ-VAE 模型结构
class VQVAE(nn.Module):
    def __init__(self, embedding_dim=720, num_embeddings=512, commitment_cost=0.25):
        super(VQVAE, self).__init__()
        self.d_model = embedding_dim

        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),  # 96x96x32
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), # 48x48x64
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 24x24x128
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, embedding_dim, kernel_size=4, stride=2, padding=1),  # 12x12xembedding_dim
        )
        
        # 向量量化层
        self.vq_layer = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)

        # 解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embedding_dim, 128, kernel_size=4, stride=2, padding=1), # 24
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), # 48
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 96
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),   # 192
            nn.Sigmoid()
        )

    def forward(self, x):
        # 编码
        z = self.encoder(x)
        # 向量量化
        quantized, vq_loss = self.vq_layer(z)
        # 解码
        x_recon = self.decoder(quantized)
        return x_recon, vq_loss
