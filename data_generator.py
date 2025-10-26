import numpy as np 
import pandas as pd
import torch
import matplotlib.pyplot as plt
import random
import os
from Model import Generator


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NOISE_DIM = int(os.environ.get('NOISE_DIM', 128))  # default if not set
SAVE_DIR = "Synthetic_data"
IMG_DIR = "images"

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)


def generate_samples(generator, class_label, num_samples):
    """Generate synthetic signals using the trained Generator."""
    generator.eval()
    generated_samples = []

    with torch.no_grad():
        for _ in range(num_samples):
            noise = torch.randn(1, NOISE_DIM).to(DEVICE)
            label_tensor = torch.tensor([class_label]).to(DEVICE)
            generated = generator(noise, label_tensor)  # Shape: (1, seq_len, 1)
            generated_samples.append(generated.cpu().numpy()[0, :, 0])  # Flatten

    generator.train()
    return np.array(generated_samples)  # Shape: (num_samples, seq_len)


def visualize_random_samples(samples, class_label,model_type, num_to_show=2):
    """Randomly visualize a few generated samples and save plots."""
    if len(samples) == 0:
        print("No samples available for visualization.")
        return

    class_name = "No Disease" if class_label == 0 else "Disease"
    random_indices = random.sample(range(len(samples)), min(num_to_show, len(samples)))
    selected_samples = samples[random_indices]

    fig, axes = plt.subplots(1, len(selected_samples), figsize=(10, 4))
    if len(selected_samples) == 1:
        axes = [axes]

    for i, sample in enumerate(selected_samples):
        axes[i].plot(sample, linewidth=0.8)
        axes[i].set_title(f'Sample {random_indices[i] + 1}')
        axes[i].set_xlabel('Time Steps')
        axes[i].set_ylabel('Amplitude')
        axes[i].grid(True, alpha=0.3)

    plt.suptitle(f'Synthetic Data using {model_type}for - Class {class_label} ({class_name})', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{IMG_DIR}/synthetic_data_class_{class_label}_{model_type}.png', dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    print("\n" + "="*60)
    print("Generate Synthetic Data")
    print("="*60)

    # Load trained generator
    model = Generator().to(DEVICE)
    cgan_path='Trained_model/cGAN_generator_20251026132044.pth'
    cwgan_path='Trained_model/cGAN_generator_20251026132044.pth'
    model.load_state_dict(torch.load(cgan_path, map_location=DEVICE))
    generator_to_use = model

    while True:
        try:
            class_input = input("\nEnter class (0 for No Disease, 1 for Disease, or 'q' to quit): ")
            if class_input.lower() == 'q':
                break

            class_label = int(class_input)
            if class_label not in [0, 1]:
                print("Invalid class! Please enter 0 or 1.")
                continue

            num_samples = int(input("Enter number of samples to generate: "))
            if num_samples <= 0:
                print("Number of samples must be positive! Try again.")
                continue

            # Generate data
            generated_data = generate_samples(generator_to_use, class_label, num_samples)
            print(f"\nGenerated {num_samples} samples for class {class_label}")
            print(f"Shape of generated data: {generated_data.shape}")

            signal_df = pd.DataFrame(generated_data)
            signal_df['label'] = class_label
            signal_df['severity'] = 5

            csv_filename = f'{SAVE_DIR}/class_{class_label}.csv'
            signal_df.to_csv(csv_filename, index=False)
            print(f"\n signals saved to '{csv_filename}'")

            # Visualize random samples
            visualize_random_samples(generated_data, class_label,'cWGAN', num_to_show=2)

        except ValueError:
            print("Invalid input! Please enter valid numbers.")
        except KeyboardInterrupt:
            print("\nExiting...")
            break
