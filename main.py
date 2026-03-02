import sys
import yaml
import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

def main():
    try:
        import torch
        from models.unet import Unet3D
        from diffusion import GaussianDiffusion
        from dataloader import UCF101Dataset
    except ImportError as e:
        print("Error: Required libraries not found.")
        print(f"Details: {e}")
        print("Please install the requirements using:")
        print("pip install -r requirements.txt")
        return

    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create output directories
    os.makedirs(config['log_dir'], exist_ok=True)
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['sample_dir'], exist_ok=True)
    
    # Copy important files to log directory for record keeping
    import shutil
    files_to_copy = ['config.yaml', 'main.py', 'diffusion.py', 'unet.py', 'dataloader.py']
    
    for file in files_to_copy:
        if os.path.exists(file):
            dest_path = os.path.join(config['log_dir'], file)
            shutil.copy2(file, dest_path)
            print(f"Copied {file} to {dest_path}")

    # Initialize 3D U-Net
    print("Initializing 3D U-Net...")
    model = Unet3D(
        dim=config['dim'],
        dim_mults=config['dim_mults'],
        channels=config['channels'],
        time_steps=config['time_steps']
    ).to(device)

    print("Model initialized.")

    # Initialize Diffusion
    print("Initializing Gaussian Diffusion...")
    diffusion = GaussianDiffusion(
        model,
        image_size=config['image_size'],
        timesteps=config['timesteps'],
        objective=config['objective']
    ).to(device)

    print("Diffusion initialized.")

    # Initialize DataLoaders
    print("\n--- Initializing DataLoaders ---")
    
    # Transform pipeline
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((config['image_size'], config['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Train DataLoader
    train_dataset = UCF101Dataset(
        data_path=config['data_path'],
        image_size=config['image_size'],
        time_steps=config['time_steps'],
        transform=transform,
        split_file=config['train_split_path']
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4
    )
    print(f"Train DataLoader initialized with {len(train_loader.dataset)} videos.")
    
    # Test DataLoader
    test_dataset = UCF101Dataset(
        data_path=config['data_path'],
        image_size=config['image_size'],
        time_steps=config['time_steps'],
        transform=transform,
        split_file=config['test_split_path']
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4
    )
    print(f"Test DataLoader initialized with {len(test_loader.dataset)} videos.")

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=config['log_dir'])
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config['learning_rate']),
        weight_decay=float(config['weight_decay'])
    )

    # Training loop
    print("\n--- Starting Training ---")
    for epoch in range(config['epochs']):
        model.train()
        total_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}")
        for batch in pbar:
            batch = batch.to(device)
            
            # Random timesteps
            t = torch.randint(0, config['timesteps'], (batch.size(0),)).to(device)
            
            # Compute loss
            loss = diffusion.p_losses(batch, t)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
        
        # Average loss
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{config['epochs']}, Average Loss: {avg_loss:.6f}")
        
        # Log loss to TensorBoard
        writer.add_scalar('Loss/Average', avg_loss, epoch+1)
        writer.add_scalar('Loss/Learning_Rate', optimizer.param_groups[0]['lr'], epoch+1)
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(config['checkpoint_dir'], f"checkpoint_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
            
            # Log checkpoint event
            writer.add_text('Checkpoint', f"Saved at epoch {epoch+1}", epoch+1)

    print("\n--- Training Complete ---")
    
    # Close TensorBoard writer
    writer.close()
    
    print("Done.")

if __name__ == "__main__":
    main()
