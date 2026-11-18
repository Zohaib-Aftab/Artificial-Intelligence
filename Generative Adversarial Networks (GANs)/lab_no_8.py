"""Task No 1
Use a GAN in PyTorch to generate handwritten digit images from random noise using Fashion MNIST dataset (handwritten digits)."""
!pip install -q torch torchvision matplotlib numpy torchmetrics torch-fidelity
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import FashionMNIST
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from google.colab import drive
drive.mount('/content/drive')
save_dir = "/content/drive/MyDrive/FashionMNIST_GAN"
checkpoint_dir = os.path.join(save_dir, "checkpoints")
image_dir = os.path.join(save_dir, "generated_images")
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(image_dir, exist_ok=True)
checkpoint_path = os.path.join(
    checkpoint_dir,
    "gan_checkpoint.pth"
)
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
print("Using device:", device)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_dataset = FashionMNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

batch_size = 128
data_rng = torch.Generator()
data_rng.manual_seed(42)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    generator=data_rng
)
print("Total training images:", len(train_dataset))
images, labels = next(iter(train_loader))
fig, axes = plt.subplots(2, 5, figsize=(10, 5))
for i, ax in enumerate(axes.flat):
    image = images[i].squeeze().numpy()
    image = (image + 1) / 2
    ax.imshow(image, cmap="gray")
    ax.set_title(f"Label: {labels[i].item()}")
    ax.axis("off")
plt.tight_layout()
plt.show()
class Generator(nn.Module):
    def __init__(self, noise_dim=100):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 784),
            nn.Tanh()
        )
    def forward(self, z):
        image = self.model(z)
        image = image.view(-1, 1, 28, 28)
        return image
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    def forward(self, image):
        return self.model(image)
noise_dim = 100
generator = Generator(noise_dim).to(device)
discriminator = Discriminator().to(device)
criterion = nn.BCELoss()
optimizer_G = optim.Adam(
    generator.parameters(),
    lr=0.0002,
    betas=(0.5, 0.999)
)
optimizer_D = optim.Adam(
    discriminator.parameters(),
    lr=0.0002,
    betas=(0.5, 0.999)
)
print(generator)
print(discriminator)
epochs = 100
checkpoint_interval = 10
G_losses = []
D_losses = []
fixed_noise = torch.randn(
    64,
    noise_dim,
    device=device
)
start_epoch = 0
if os.path.exists(checkpoint_path):
    print("\nLoading previous checkpoint...")
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False
    )
    generator.load_state_dict(
        checkpoint["generator"]
    )
    discriminator.load_state_dict(
        checkpoint["discriminator"]
    )
    optimizer_G.load_state_dict(
        checkpoint["optimizer_G"]
    )
    optimizer_D.load_state_dict(
        checkpoint["optimizer_D"]
    )
    start_epoch = checkpoint["epoch"]
    G_losses = checkpoint["G_losses"]
    D_losses = checkpoint["D_losses"]
    fixed_noise = checkpoint["fixed_noise"].to(device)
    data_rng.set_state(
        checkpoint["data_rng_state"]
    )
    torch.set_rng_state(
        checkpoint["torch_rng_state"]
    )
    if torch.cuda.is_available() and "cuda_rng_state" in checkpoint:
        torch.cuda.set_rng_state_all(
            checkpoint["cuda_rng_state"]
        )
    print("Checkpoint loaded successfully!")
    print("Previous completed epoch:", start_epoch)
    print("Resuming from epoch:", start_epoch + 1)
else:
    print("\nStarting new GAN training...")
for epoch in range(start_epoch, epochs):
    generator.train()
    discriminator.train()
    total_G_loss = 0
    total_D_loss = 0
    for real_images, _ in train_loader:
        real_images = real_images.to(device)
        current_batch_size = real_images.size(0)
        real_labels = torch.ones(
            current_batch_size, 1, device=device
        )
        fake_labels = torch.zeros(
            current_batch_size, 1, device=device
        )
        optimizer_D.zero_grad()
        real_output = discriminator(real_images)
        real_loss = criterion(
            real_output,
            real_labels
        )
        noise = torch.randn(
            current_batch_size,
            noise_dim,
            device=device
        )
        fake_images = generator(noise)
        fake_output = discriminator(
            fake_images.detach()
        )
        fake_loss = criterion(
            fake_output,
            fake_labels
        )
        D_loss = real_loss + fake_loss
        D_loss.backward()
        optimizer_D.step()
        optimizer_G.zero_grad()
        noise = torch.randn(
            current_batch_size,
            noise_dim,
            device=device
        )
        generated_images = generator(noise)
        output = discriminator(
            generated_images
        )
        G_loss = criterion(
            output,
            real_labels
        )
        G_loss.backward()
        optimizer_G.step()
        total_D_loss += D_loss.item()
        total_G_loss += G_loss.item()
    avg_D_loss = total_D_loss / len(train_loader)
    avg_G_loss = total_G_loss / len(train_loader)
    D_losses.append(avg_D_loss)
    G_losses.append(avg_G_loss)
    print(
        f"Epoch [{epoch + 1}/{epochs}] "
        f"D Loss: {avg_D_loss:.4f} "
        f"G Loss: {avg_G_loss:.4f}"
    )
    if (epoch + 1) % checkpoint_interval == 0:
        generator.eval()
        with torch.no_grad():
            sample_images = generator(fixed_noise)
            save_image(
                sample_images,
                os.path.join(
                    image_dir,
                    f"epoch_{epoch + 1}.png"
                ),
                nrow=8,
                normalize=True,
                value_range=(-1, 1)
            )
        print(
            f"Generated images saved at epoch {epoch + 1}"
        )
    checkpoint = {
        "epoch": epoch + 1,
        "generator": generator.state_dict(),
        "discriminator": discriminator.state_dict(),
        "optimizer_G": optimizer_G.state_dict(),
        "optimizer_D": optimizer_D.state_dict(),
        "G_losses": G_losses,
        "D_losses": D_losses,
        "fixed_noise": fixed_noise,
        "data_rng_state": data_rng.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None
        )
    }
    torch.save(
        checkpoint,
        checkpoint_path
    )
    if (epoch + 1) % checkpoint_interval == 0:
        snapshot_path = os.path.join(
            checkpoint_dir,
            f"checkpoint_epoch_{epoch + 1}.pth"
        )
        torch.save(
            checkpoint,
            snapshot_path
        )
        print(
            f"Checkpoint snapshot saved at epoch {epoch + 1}"
        )
torch.save(
    generator.state_dict(),
    os.path.join(save_dir, "generator_final.pth")
)
print("\nGAN Training Completed!")
plt.figure(figsize=(10, 5))
plt.plot(
    G_losses,
    label="Generator Loss"
)
plt.plot(
    D_losses,
    label="Discriminator Loss"
)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("GAN Training Loss")
plt.legend()
plt.grid(True)
plt.savefig(
    os.path.join(save_dir, "training_loss.png")
)
plt.show()
generator.eval()
with torch.no_grad():
    noise = torch.randn(
        64,
        noise_dim,
        device=device
    )
    generated_images = generator(noise)
grid = make_grid(
    generated_images,
    nrow=8,
    normalize=True,
    value_range=(-1, 1)
)
plt.figure(figsize=(8, 8))
plt.imshow(
    grid.permute(1, 2, 0).cpu()
)
plt.axis("off")
plt.title("Generated Fashion-MNIST Images")
plt.show()
save_image(
    generated_images,
    os.path.join(
        save_dir,
        "generated_fashion_mnist.png"
    ),
    nrow=8,
    normalize=True,
    value_range=(-1, 1)
)
print("Generated images saved successfully!")
fid = FrechetInceptionDistance(
    feature=2048,
    normalize=False
).to(device)
generator.eval()
num_images = 10000
generated_count = 0
real_count = 0
for real_images, _ in train_loader:
    real_images = real_images.to(device)
    remaining = num_images - real_count
    real_images = real_images[:remaining]
    real_images = (
        (real_images + 1) / 2 * 255
    )
    real_images = real_images.clamp(
        0, 255
    ).to(torch.uint8)
    real_images = real_images.repeat(
        1, 3, 1, 1
    )
    fid.update(
        real_images,
        real=True
    )
    real_count += real_images.size(0)
    if real_count >= num_images:
        break
with torch.no_grad():
    while generated_count < num_images:
        current_batch = min(
            128,
            num_images - generated_count
        )
        noise = torch.randn(
            current_batch,
            noise_dim,
            device=device
        )
        fake_images = generator(noise)
        fake_images = (
            (fake_images + 1) / 2 * 255
        )
        fake_images = fake_images.clamp(
            0, 255
        ).to(torch.uint8)
        fake_images = fake_images.repeat(
            1, 3, 1, 1
        )
        fid.update(
            fake_images,
            real=False
        )
        generated_count += current_batch
fid_score = fid.compute()
print("FID Score:", fid_score.item())
inception_score = InceptionScore(
    normalize=False,
    splits=10
).to(device)
generator.eval()
num_images = 10000
generated_count = 0
with torch.no_grad():
    while generated_count < num_images:
        current_batch = min(
            128,
            num_images - generated_count
        )
        noise = torch.randn(
            current_batch,
            noise_dim,
            device=device
        )
        fake_images = generator(noise)
        fake_images = (
            (fake_images + 1) / 2 * 255
        )
        fake_images = fake_images.clamp(
            0, 255
        ).to(torch.uint8)
        fake_images = fake_images.repeat(
            1, 3, 1, 1
        )
        inception_score.update(fake_images)
        generated_count += current_batch
is_mean, is_std = inception_score.compute()
print("Inception Score:", is_mean.item())
print("Standard Deviation:", is_std.item())
print("=" * 50)
print("GAN PERFORMANCE EVALUATION")
print("=" * 50)
print(f"Dataset          : Fashion-MNIST")
print(f"Training Epochs  : {epochs}")
print(f"Noise Dimension  : {noise_dim}")
print(f"Learning Rate    : 0.0002")
print(f"Batch Size       : {batch_size}")
print("-" * 50)
print(f"FID Score        : {fid_score.item():.4f}")
print(f"Inception Score  : {is_mean.item():.4f}")
print(f"IS Std Deviation : {is_std.item():.4f}")
print("=" * 50)
print("\nAll project files saved in:")
print(save_dir)
print("\nCheckpoint folder:")
print(checkpoint_dir)
print("\nGenerated images folder:")
print(image_dir)

"""Task No 2
Use a conditional GAN in PyTorch to translate grayscale images into edge maps using CIFAR-10 images converted to grayscale with edge maps created using the Sobel filter."""
!pip install -q torch torchvision matplotlib numpy
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10
from torchvision.utils import make_grid, save_image
from google.colab import drive
drive.mount('/content/drive')
save_dir = "/content/drive/MyDrive/CIFAR10_CGAN"
checkpoint_dir = os.path.join(save_dir, "checkpoints")
image_dir = os.path.join(save_dir, "generated_images")
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(image_dir, exist_ok=True)
checkpoint_path = os.path.join(checkpoint_dir, "cgan_checkpoint.pth")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
transform = transforms.ToTensor()
cifar_dataset = CIFAR10(root="./data", train=True,
                        download=True, transform=transform)
class GrayscaleCIFAR10(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, index):
        image, _ = self.dataset[index]
        grayscale = transforms.functional.rgb_to_grayscale(image)
        return grayscale
train_dataset = GrayscaleCIFAR10(cifar_dataset)
batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                          num_workers=2, pin_memory=torch.cuda.is_available(), drop_last=True)
print("Total training images:", len(train_dataset))
def sobel_edge_detection(images):
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                           dtype=torch.float32,
                           device=images.device).view(1, 1, 3, 3) / 4.0
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                           dtype=torch.float32,
                           device=images.device).view(1, 1, 3, 3) / 4.0
    gradient_x = F.conv2d(images, sobel_x, padding=1)
    gradient_y = F.conv2d(images, sobel_y, padding=1)
    magnitude = torch.sqrt(gradient_x ** 2 + gradient_y ** 2 + 1e-8)
    maximum = magnitude.amax(dim=(2, 3), keepdim=True).clamp(min=1e-6)
    edges = magnitude / maximum
    return edges.clamp(0, 1)
sample_images, _ = next(iter(DataLoader(cifar_dataset, batch_size=8)))
gray_images = torch.stack([transforms.functional.rgb_to_grayscale(img) for img in sample_images])
edge_images = sobel_edge_detection(gray_images)
fig, axes = plt.subplots(3, 5, figsize=(12, 7))
for i in range(5):
    axes[0, i].imshow(sample_images[i].permute(1, 2, 0))
    axes[0, i].set_title("Original")
    axes[1, i].imshow(gray_images[i].squeeze(), cmap="gray")
    axes[1, i].set_title("Grayscale")
    axes[2, i].imshow(edge_images[i].squeeze(), cmap="gray")
    axes[2, i].set_title("Sobel Edge")
    for j in range(3):
        axes[j, i].axis("off")
plt.tight_layout()
plt.show()
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.down1 = nn.Sequential(nn.Conv2d(1, 64, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True))
        self.down2 = nn.Sequential(nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True))
        self.down3 = nn.Sequential(nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True))
        self.down4 = nn.Sequential(nn.Conv2d(256, 512, 4, 2, 1), nn.BatchNorm2d(512), nn.LeakyReLU(0.2, inplace=True))
        self.down5 = nn.Sequential(nn.Conv2d(512, 512, 4, 2, 1), nn.BatchNorm2d(512), nn.LeakyReLU(0.2, inplace=True))
        self.up1 = nn.Sequential(nn.ConvTranspose2d(512, 512, 4, 2, 1), nn.BatchNorm2d(512), nn.ReLU(inplace=True), nn.Dropout(0.5))
        self.up2 = nn.Sequential(nn.ConvTranspose2d(1024, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.up3 = nn.Sequential(nn.ConvTranspose2d(512, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.up4 = nn.Sequential(nn.ConvTranspose2d(256, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.up5 = nn.Sequential(nn.ConvTranspose2d(128, 1, 4, 2, 1), nn.Tanh())
    def forward(self, image):
        d1 = self.down1(image)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        u1 = self.up1(d5)
        u1 = torch.cat([u1, d4], dim=1)
        u2 = self.up2(u1)
        u2 = torch.cat([u2, d3], dim=1)
        u3 = self.up3(u2)
        u3 = torch.cat([u3, d2], dim=1)
        u4 = self.up4(u3)
        u4 = torch.cat([u4, d1], dim=1)
        output = self.up5(u4)
        return output
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(2, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 1, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 3, 1, 1)
        )
    def forward(self, grayscale, edge):
        combined = torch.cat([grayscale, edge], dim=1)
        return self.model(combined)
generator = Generator().to(device)
discriminator = Discriminator().to(device)
print(generator)
print(discriminator)
adversarial_loss = nn.BCEWithLogitsLoss()
pixel_loss = nn.L1Loss()
LR = 0.0002
BETA_1 = 0.5
LAMBDA_L1 = 100
optimizer_G = optim.Adam(generator.parameters(), lr=LR, betas=(BETA_1, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=LR, betas=(BETA_1, 0.999))
epochs = 100
checkpoint_interval = 10
G_losses = []
D_losses = []
start_epoch = 0
if os.path.exists(checkpoint_path):
    print("Loading previous checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    generator.load_state_dict(checkpoint["generator"])
    discriminator.load_state_dict(checkpoint["discriminator"])
    optimizer_G.load_state_dict(checkpoint["optimizer_G"])
    optimizer_D.load_state_dict(checkpoint["optimizer_D"])
    start_epoch = checkpoint["epoch"]
    G_losses = checkpoint["G_losses"]
    D_losses = checkpoint["D_losses"]
    print("Checkpoint loaded successfully!")
    print("Starting from epoch:", start_epoch + 1)
else:
    print("Starting new CGAN training...")
fixed_images = next(iter(train_loader))[:8].to(device)
fixed_edges = sobel_edge_detection(fixed_images)
fixed_gray = fixed_images * 2 - 1
fixed_edges = fixed_edges * 2 - 1
for epoch in range(start_epoch, epochs):
    generator.train()
    discriminator.train()
    total_G_loss = 0
    total_D_loss = 0
    for grayscale in train_loader:
        grayscale = grayscale.to(device)
        current_batch = grayscale.size(0)
        real_edges = sobel_edge_detection(grayscale)
        gray_input = grayscale * 2 - 1
        real_edges = real_edges * 2 - 1
        optimizer_D.zero_grad()
        fake_edges = generator(gray_input)
        real_output = discriminator(gray_input, real_edges)
        fake_output = discriminator(gray_input, fake_edges.detach())
        real_loss = adversarial_loss(real_output, torch.ones_like(real_output))
        fake_loss = adversarial_loss(fake_output, torch.zeros_like(fake_output))
        D_loss = (real_loss + fake_loss) / 2
        D_loss.backward()
        optimizer_D.step()
        optimizer_G.zero_grad()
        generated_edges = generator(gray_input)
        fake_output = discriminator(gray_input, generated_edges)
        G_adv_loss = adversarial_loss(fake_output, torch.ones_like(fake_output))
        G_pixel_loss = pixel_loss(generated_edges, real_edges)
        G_loss = G_adv_loss + LAMBDA_L1 * G_pixel_loss
        G_loss.backward()
        optimizer_G.step()
        total_D_loss += D_loss.item()
        total_G_loss += G_loss.item()
    avg_D_loss = total_D_loss / len(train_loader)
    avg_G_loss = total_G_loss / len(train_loader)
    D_losses.append(avg_D_loss)
    G_losses.append(avg_G_loss)
    print(f"Epoch [{epoch+1}/{epochs}] D Loss: {avg_D_loss:.4f} G Loss: {avg_G_loss:.4f}")
    if (epoch + 1) % checkpoint_interval == 0:
        generator.eval()
        with torch.no_grad():
            generated = generator(fixed_gray)
            comparison = torch.cat([fixed_gray, fixed_edges, generated], dim=0)
            grid = make_grid(comparison, nrow=8, normalize=True, value_range=(-1, 1))
            save_image(grid, os.path.join(image_dir, f"epoch_{epoch+1}.png"))
        print("Generated images saved:", epoch + 1)
    checkpoint = {
        "epoch": epoch + 1,
        "generator": generator.state_dict(),
        "discriminator": discriminator.state_dict(),
        "optimizer_G": optimizer_G.state_dict(),
        "optimizer_D": optimizer_D.state_dict(),
        "G_losses": G_losses,
        "D_losses": D_losses
    }
    torch.save(checkpoint, checkpoint_path)
    if (epoch + 1) % checkpoint_interval == 0:
        torch.save(checkpoint, os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth"))
        print("Checkpoint saved:", epoch + 1)
torch.save(generator.state_dict(), os.path.join(save_dir, "generator_final.pth"))
print("CGAN Training Completed!")
plt.figure(figsize=(10, 5))
plt.plot(G_losses, label="Generator Loss")
plt.plot(D_losses, label="Discriminator Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("CGAN Training Loss")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(save_dir, "training_loss.png"))
plt.show()
generator.eval()
with torch.no_grad():
    final_edges = generator(fixed_gray)
fig, axes = plt.subplots(3, 8, figsize=(16, 7))
for i in range(8):
    axes[0, i].imshow(fixed_gray[i, 0].cpu() * 0.5 + 0.5, cmap="gray")
    axes[0, i].set_title("Grayscale")
    axes[1, i].imshow(fixed_edges[i, 0].cpu() * 0.5 + 0.5, cmap="gray")
    axes[1, i].set_title("Real Edge")
    axes[2, i].imshow(final_edges[i, 0].cpu() * 0.5 + 0.5, cmap="gray")
    axes[2, i].set_title("Generated Edge")
    for j in range(3):
        axes[j, i].axis("off")
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "final_comparison.png"))
plt.show()
print("=" * 50)
print("CONDITIONAL GAN PERFORMANCE REPORT")
print("=" * 50)
print("Dataset          : CIFAR-10")
print("Input            : Grayscale Images")
print("Target           : Sobel Edge Maps")
print("Framework        : PyTorch")
print("GAN Architecture : U-Net + PatchGAN")
print("Training Epochs  :", epochs)
print("Batch Size       :", batch_size)
print("Learning Rate    :", LR)
print("L1 Loss Weight   :", LAMBDA_L1)
print("-" * 50)
print("Final Generator Loss:", G_losses[-1])
print("Final Discriminator Loss:", D_losses[-1])
print("=" * 50)
print("All project files saved in:")
print(save_dir)