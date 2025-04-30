import torch
import os
from torchvision.utils import save_image
from model import Generator

# Hyperparameters (should match training)
latent_dim = 100
img_channels = 3
image_size = 128  # Images were resized to 128x128 during training

# Paths
generator_weights = r"C:\Users\Vijay Patidar\Desktop\hq_dataset\outputs\models\generator_epoch_55.pth"
output_dir = r"C:\Users\Vijay Patidar\Desktop\hq_dataset\outputs\generated_samples"
os.makedirs(output_dir, exist_ok=True)

# Select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load Generator
generator = Generator(latent_dim, img_channels).to(device)
generator.load_state_dict(torch.load(generator_weights, map_location=device))
generator.eval()

# Generate images
num_images = 5  # You can change this number to generate more images
with torch.no_grad():
    z = torch.randn(num_images, latent_dim).to(device)
    fake_imgs = generator(z)

    for idx, img in enumerate(fake_imgs):
        save_path = os.path.join(output_dir, f"generated_image_{idx+1}.png")
        save_image(img, save_path, normalize=True)
        print(f"Image saved at: {save_path}")


# # import torch
# # import os
# # from torchvision.utils import save_image
# # from model import Generator
# # from PIL import Image  # No need for Resampling

# # # Hyperparameters
# # latent_dim = 100
# # img_channels = 3
# # output_image_size = (4, 4)  # Desired output resolution (e.g., 64x64)

# # # Paths
# # model_path = r"C:\Users\Vijay Patidar\Desktop\hq_dataset\outputs\models"
# # output_path = r"C:\Users\Vijay Patidar\Desktop\hq_dataset\outputs\generated_samples"
# # os.makedirs(output_path, exist_ok=True)

# # # Select device
# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # print(f"Using device: {device}")

# # # Specify the epoch to load
# # epoch_to_load = 22  # Change to the desired epoch number
# # model_name = f"generator_epoch_{epoch_to_load}.pth"
# # model_path_full = os.path.join(model_path, model_name)

# # if not os.path.exists(model_path_full):
# #     raise RuntimeError(f"Model for epoch {epoch_to_load} not found at {model_path_full}")

# # print(f"Loading model: {model_path_full}")

# # # Load Generator
# # generator = Generator(latent_dim, img_channels).to(device)
# # generator.load_state_dict(torch.load(model_path_full, map_location=device))
# # generator.eval()

# # # Generate an image
# # with torch.no_grad():
# #     z = torch.randn(1, latent_dim).to(device)  # Generate a random latent vector
# #     fake_img = generator(z)
# #     fake_img_path = os.path.join(output_path, f"generated_epoch_{epoch_to_load}.png")
    
# #     # Save the original generated image
# #     save_image(fake_img.data, fake_img_path, normalize=True)

# #     # Resize the generated image to the desired resolution
# #     img = Image.open(fake_img_path)
# #     img_resized = img.resize(output_image_size, Image.LANCZOS)  # Use Image.LANCZOS
# #     resized_path = os.path.join(output_path, f"generated_epoch_{epoch_to_load}_resized.png")
# #     img_resized.save(resized_path)

# #     print(f"Original image saved at: {fake_img_path}")
# #     print(f"Resized image saved at: {resized_path}")

