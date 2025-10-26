import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import torch
from dotenv import load_dotenv
from Model.GAN.model import Generator,Discriminator
from Model.GAN.cGAN import cGAN
from Dataset.dataset import train_loader
from datetime import datetime
load_dotenv()
device=os.environ.get('DEVICE','cpu') 
noise_dim=int(os.environ['NOISE_DIM'])
# Training Loop
def train_gan(model, train_loader, epochs, model_type='cGAN'):
    g_losses = []
    d_losses = []

    print(f"\nTraining {model_type}...")
    print(f"Total batches per epoch: {len(train_loader)}")

    for epoch in range(epochs):
        epoch_g_loss = 0
        epoch_d_loss = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for real_data, labels in pbar:
            real_data = real_data.to(device).float()
            labels = labels.to(device)

            # Train step
            d_loss, g_loss = model.train_step(real_data, labels)

            epoch_d_loss += d_loss
            epoch_g_loss += g_loss

            pbar.set_postfix({
                'D_loss': f'{d_loss:.4f}',
                'G_loss': f'{g_loss:.4f}'
            })

        # Average losses
        avg_d_loss = epoch_d_loss / len(train_loader)
        avg_g_loss = epoch_g_loss / len(train_loader)

        g_losses.append(avg_g_loss)
        d_losses.append(avg_d_loss)

        print(f"Epoch [{epoch+1}/{epochs}] | D_loss: {avg_d_loss:.4f} | G_loss: {avg_g_loss:.4f}")

        # Save samples every 10 epochs
        if (epoch + 1) % 10 == 0:
            visualize_samples(model.generator, epoch + 1, model_type)

    # Plot training losses
    plot_losses(g_losses, d_losses, model_type)

    return model

# Visualization Functions
def visualize_samples(generator, epoch, model_type, num_samples=4):
    """Generate and visualize samples"""
    generator.eval()

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    with torch.no_grad():
        for i, label in enumerate([0, 0, 1, 1]):
            noise = torch.randn(1, noise_dim|128).to(device)
            label_tensor = torch.tensor([label]).to(device)

            generated = generator(noise, label_tensor)
            generated = generated.cpu().numpy()[0, :, 0]

            axes[i].plot(generated)
            axes[i].set_title(f'Class {label} ({"No Disease" if label == 0 else "Disease"})')
            axes[i].set_xlabel('Time Steps')
            axes[i].set_ylabel('Amplitude')
            axes[i].grid(True, alpha=0.3)

    plt.suptitle(f'{model_type} - Generated Samples at Epoch {epoch}')
    plt.tight_layout()
    plt.savefig(f'sample_image/{model_type}_samples_epoch_{epoch}.png', dpi=150, bbox_inches='tight')
    plt.close()

    generator.train()

def plot_losses(g_losses, d_losses, model_type):
    """Plot training losses"""
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(g_losses, label='Generator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{model_type} - Generator Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(d_losses, label='Discriminator Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{model_type} - Discriminator Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'Images/{model_type}_training_losses.png', dpi=150, bbox_inches='tight')
    plt.close()



if __name__=='__main__':
    print("="*60)
    print("Training cGAN")
    print("="*60)

    # Initialize models
    generator_cgan = Generator()
    discriminator_cgan = Discriminator()

    # Create cGAN model
    model = cGAN(generator_cgan, discriminator_cgan, device)

    # Train cGAN
    model = train_gan(model, train_loader, 50, model_type='cGAN')

    # Save cGAN model
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    model_name = model.__class__.__name__
    torch.save(model.generator.state_dict(), f'Trained_model/{model_name}_generator_{timestamp}.pth')
    torch.save(model.discriminator.state_dict(), f'Trained_model/{model_name}_discriminator{timestamp}.pth')

