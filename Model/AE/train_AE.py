import os
from dotenv import load_dotenv
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd

from Model import (MLP_AE, CNN1D_AE, LSTM_AE)

from Dataset.dataset import train_loader,val_loader,test_loader
load_dotenv()

seq_len=int(os.environ['WINDOW_SIZE'])
in_channel=int(os.environ['NUM_VARIABLES'])
num_classes=int(os.environ['NUM_CLASSES'])


def train_ae(model, train_loader, val_loader, test_loader, lr=0.001, epochs=80, device='cpu'):
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    best_model_state = None

    train_losses = []
    val_losses = []

    print(f"Training {model.__class__.__name__}...")

    for epoch in range(epochs):
   
        # Training phase
        model.train()
        train_loss = 0
        for x, _ in train_loader:
            x = x.to(device).float()
            optimizer.zero_grad()
            x_recon = model(x)
            loss = criterion(x_recon, x)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

 
        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, _ in val_loader:
                x = x.to(device).float()
                x_recon = model(x)
                loss = criterion(x_recon, x)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

 
    # Test phase
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device).float()
            x_recon = model(x)
            loss = criterion(x_recon, x)
            test_loss += loss.item()
    avg_test_loss = test_loss / len(test_loader)
    print(f"\nTest Reconstruction Loss: {avg_test_loss:.6f}")


    # training curve
    plt.figure(figsize=(8,5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("MSE Loss")
    plt.title(f"{model.__class__.__name__} Training Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    model_name = model.__class__.__name__
    plt.savefig(f'Images/{model_name}_training_losses.png', dpi=150, bbox_inches='tight')
    plt.close()

    return model, train_losses, val_losses,avg_test_loss


if __name__=='__main__':
    ae1 = MLP_AE(seq_len * 1)
    ae2 = CNN1D_AE(1)
    ae3 = LSTM_AE(seq_len)
    results = []
    for model in [ae1, ae2, ae3]:
        model, train_losses, val_losses, test_loss = train_ae(model, train_loader, val_loader, test_loader, lr=1e-3, epochs=30)
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        model_name = model.__class__.__name__
        torch.save(model.state_dict(), f'Trained_model/{model_name}_{timestamp}.pth')

        results.append({
            'Model Name': model_name,
            'Epochs': 30, 
            'Final Train Loss': train_losses[-1],
            'Final Validation Loss': val_losses[-1],
            'Test Reconstruction Loss': test_loss,
        })

    #results in a table and save to a markdown file
    results_df = pd.DataFrame(results)
    markdown_content = (
    "# Autoencoder Training Results\n\n"
    "In this page, the training results of several Autoencoders are shown. "
    "The Autoencoders are defined in directory [AE](AE).\n\n"
    + results_df.to_markdown(index=False)
)

    # Save results to a markdown file
    with open('training_AEs_results.md', 'w') as f:
        
        f.write(markdown_content)

    print("\nTraining results saved to training_AEs_results.md")