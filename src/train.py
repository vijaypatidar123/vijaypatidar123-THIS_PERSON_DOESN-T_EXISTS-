 
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from dataset import ImageDataset
from model import Generator, Discriminator
import torchvision.transforms as transforms
import os

# Hyperparameters
latent_dim = 100
batch_size = 64
learning_rate = 0.0002
epochs = 1000
img_channels = 3

# Paths
dataset_path = r"C:\Users\Vijay Patidar\Desktop\hq_dataset\data\images"
output_path = r"C:\Users\Vijay Patidar\Desktop\hq_dataset\outputs\generated_samples"
model_path = r"C:\Users\Vijay Patidar\Desktop\hq_dataset\outputs\models"

os.makedirs(output_path, exist_ok=True)
os.makedirs(model_path, exist_ok=True)

# Select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load Dataset
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
])
dataset = ImageDataset(root_dir=dataset_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize Models
generator = Generator(latent_dim, img_channels).to(device)
discriminator = Discriminator(img_channels).to(device)

# Loss Function and Optimizers
adversarial_loss = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

# Training Loop
for epoch in range(epochs):
    for batch_idx, real_imgs in enumerate(dataloader):
        real_imgs = real_imgs.to(device)

        # Labels
        real_labels = torch.ones(real_imgs.size(0), 1).to(device)
        fake_labels = torch.zeros(real_imgs.size(0), 1).to(device)

        # ---------------------
        # Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()

        # Real Images
        real_loss = adversarial_loss(discriminator(real_imgs), real_labels)

        # Fake Images
        z = torch.randn(real_imgs.size(0), latent_dim).to(device)
        fake_imgs = generator(z)
        fake_loss = adversarial_loss(discriminator(fake_imgs.detach()), fake_labels)

        # Total Discriminator Loss
        d_loss = real_loss + fake_loss
        d_loss.backward()
        optimizer_D.step()

        # -----------------
        # Train Generator
        # -----------------
        optimizer_G.zero_grad()

        # Fake Images
        g_loss = adversarial_loss(discriminator(fake_imgs), real_labels)  # Trick discriminator
        g_loss.backward()
        optimizer_G.step()

        # Logging
        if batch_idx % 100 == 0:
            print(f"Epoch [{epoch}/{epochs}] Batch {batch_idx} Loss D: {d_loss.item()}, Loss G: {g_loss.item()}")

            # Save Generated Samples
            save_image(fake_imgs.data[:25], os.path.join(output_path, f"{epoch}_{batch_idx}.png"), nrow=5, normalize=True)

    # Save model weights after each epoch
    torch.save(generator.state_dict(), os.path.join(model_path, f"generator_epoch_{epoch}.pth"))
    torch.save(discriminator.state_dict(), os.path.join(model_path, f"discriminator_epoch_{epoch}.pth"))

print("Training complete and model weights saved.")
