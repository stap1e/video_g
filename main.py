import sys
import yaml
import os
import time, os, sys
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

def main():
    try:
        import torch
        from models.unet import Unet3D
        from models.unet import unet_3D
        from models.diffusion import GaussianDiffusion
        from utils.dataloader import UCF101Dataset
        from utils.metrics import compute_fvd, compute_fid, compute_lpips, compute_psnr, compute_ssim
        from utils.fid import extract_features_from_videos, compute_fid_from_features
        from utils.inception_score import compute_inception_score_from_sampler
    except ImportError as e:
        print("Error: Required libraries not found.")
        print(f"Details: {e}")
        print("Please install the requirements using:")
        print("pip install -r requirements.txt")
        return

    # Load configuration
    from omegaconf import OmegaConf
    config = OmegaConf.load("config.yaml")
    import datetime
    import pytz

    # 获取北京时间
    beijing_tz = pytz.timezone('Asia/Shanghai')
    beijing_time = datetime.datetime.now(beijing_tz)
    config['timestamp'] = beijing_time.strftime("%Y-%m-%d-%H:%M:%S")
    config['base_dir'] = os.path.join(config['base_dir'], config['timestamp'])
    config['log_dir'] = config['log_dir'].replace("${base_dir}", config['base_dir'])
    config['checkpoint_dir'] = config['checkpoint_dir'].replace("${base_dir}", config['base_dir'])
    config['sample_dir'] = config['sample_dir'].replace("${base_dir}", config['base_dir'])
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create output directories
    os.makedirs(config['log_dir'], exist_ok=True)
    code_save_path = os.path.join(config['log_dir'], config['code_save_dir'])
    os.makedirs(code_save_path, exist_ok=True)
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['sample_dir'], exist_ok=True)
    
    # Copy important files and directories to log directory for record keeping
    import shutil
    items_to_copy = config['items_to_save']
    
    for item in items_to_copy:
        item_type = item['type']
        item_path = item['path']
        
        if not os.path.exists(item_path):
            print(f"Warning: {item_path} does not exist, skipping...")
            continue
            
        dest_path = os.path.join(code_save_path, item_path)
        
        if item_type == 'file':
            # Ensure parent directory exists
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            shutil.copy2(item_path, dest_path)
            print(f"Copied file {item_path} to {dest_path}")
            
        elif item_type == 'dir':
            # Remove existing directory if it exists
            if os.path.exists(dest_path):
                shutil.rmtree(dest_path)
            shutil.copytree(item_path, dest_path)
            print(f"Copied directory {item_path} to {dest_path}")

    # Initialize 3D U-Net
    print("Initializing 3D U-Net...")
    model_type = config.get('model_type', 'big')
    
    if model_type == 'big':
        model = Unet3D(
            dim=config['dim'],
            dim_mults=config['dim_mults'],
            channels=config['channels'],
            time_steps=config['time_steps']
        ).to(device)
    elif model_type == 'base':
        model = unet_3D(
            in_chns=config['channels'],
            class_num=config.get('class_num', 101),
            deep=config.get('deep', False)
        ).to(device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Choose 'big' or 'base'.")

    # Count total parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Convert to million units
    total_params_m = total_params / 1_000_000
    trainable_params_m = trainable_params / 1_000_000
    non_trainable_params_m = (total_params - trainable_params) / 1_000_000
    
    # Print model parameters
    print("\n--- Model Parameters ---")
    print(f"Total parameters: {total_params_m:.3f}M")
    print(f"Trainable parameters: {trainable_params_m:.3f}M")
    print(f"Non-trainable parameters: {non_trainable_params_m:.3f}M")
    print(f"Model initialized.")
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config['learning_rate']),
        weight_decay=float(config['weight_decay'])
    )

    # Load latest weights if available
    latest_checkpoint = os.path.join(config['checkpoint_dir'], "latest.pth")
    start_epoch = 0
    
    # Best metrics tracking
    best_fvd = float('inf')
    best_fid = float('inf')
    best_epoch = 0
    
    if os.path.exists(latest_checkpoint):
        checkpoint = torch.load(latest_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        # Load best metrics if available
        if 'best_fvd' in checkpoint:
            best_fvd = checkpoint['best_fvd']
            best_fid = checkpoint['best_fid']
            best_epoch = checkpoint['best_epoch']
        print(f"Loaded latest checkpoint from epoch {start_epoch}")
        print(f"Current best metrics - FVD: {best_fvd:.6f}, FID: {best_fid:.6f} at epoch {best_epoch}")

    best_metrics_text = f"Current best metrics - FVD: {best_fvd:.6f}, FID: {best_fid:.6f} at epoch {best_epoch}"

    # Initialize Diffusion
    print(f"Initializing Gaussian Diffusion... | {best_metrics_text}")
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
    
    # Print training parameters
    print("\n--- Training Parameters ---")
    print(f"Epochs: {config['epochs']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Learning rate: {config['learning_rate']}")
    print(f"Weight decay: {config['weight_decay']}")
    print(f"Timesteps: {config['timesteps']}")
    print(f"Objective: {config['objective']}")
    
    # Training loop
    print("\n--- Starting Training ---")
    total_start_time = time.time()
    
    for epoch in range(start_epoch, config['epochs']):
        epoch_start_time = time.time()
        model.train()
        total_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}, best FVD: {best_fvd:.4f}, best FID: {best_fid:.4f}, best Epoch: {best_epoch}", ncols=180)
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
            pbar.set_postfix(
                loss=loss.item(),
                best_fvd=f"{best_fvd:.6f}",
                best_fid=f"{best_fid:.6f}",
                best_epoch=best_epoch
            )
        
        # Average loss
        avg_loss = total_loss / len(train_loader)
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        epoch_time_min = epoch_time / 60
        
        print(f"Epoch {epoch+1}/{config['epochs']}, Average Loss: {avg_loss:.6f}, Time: {epoch_time:.2f}s ({epoch_time_min:.2f}min)")
        
        # Log loss and time to TensorBoard
        writer.add_scalar('Loss/Average', avg_loss, epoch+1)
        writer.add_scalar('Loss/Learning_Rate', optimizer.param_groups[0]['lr'], epoch+1)
        writer.add_scalar('Time/Epoch', epoch_time, epoch+1)
        
        # Save latest checkpoint every epoch
        latest_checkpoint = os.path.join(config['checkpoint_dir'], "latest.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'best_fvd': best_fvd,
            'best_fid': best_fid,
            'best_epoch': best_epoch
        }, latest_checkpoint)
        print(f"Latest checkpoint saved to {latest_checkpoint}")
        
        # Validation every 5 epochs
        if (epoch + 1) % 1 == 0:
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                val_pbar = tqdm(test_loader, desc=f"Val {epoch+1}/{config['epochs']}", ncols=80)
                for batch in val_pbar:
                    batch = batch.to(device)
                    t = torch.randint(0, config['timesteps'], (batch.size(0),)).to(device)
                    loss = diffusion.p_losses(batch, t)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(test_loader)
            print(f"Epoch {epoch+1}/{config['epochs']}, Validation Loss: {avg_val_loss:.6f}")
            writer.add_scalar('Loss/Validation', avg_val_loss, epoch+1)
            
            fid_conf = config.get('fid_score', None)
            is_conf = config.get('inception_score', None)

            # Compute FVD and FID on multiple batches
            model.eval()
            num_eval_batches = 5
            fvd_scores = []
            fid_scores = []
            lpips_scores = []
            psnr_scores = []
            ssim_scores = []
            
            with torch.no_grad():
                pbar = tqdm(enumerate(test_loader), total=num_eval_batches, desc="FVD/FID Evaluation", ncols=80)
                for i, batch in pbar:
                    if i >= num_eval_batches:
                        break
                        
                    real_videos = batch.to(device)
                    sample_shape = tuple(real_videos.shape)
                    print(f"[shape-debug] real_videos={sample_shape} sample_shape={sample_shape}")
                    fake_videos = diffusion.sample(sample_shape)
                    # Generate fake videos using diffusion
                    # fake_videos = diffusion.sample(
                    #     (real_videos.size(0), config['channels'], config['time_steps'], config['image_size'], config['image_size'])
                    # )
                    
                    # Compute metrics
                    fvd_score = compute_fvd(real_videos, fake_videos, device=device)
                    real_frames = real_videos[:, :, 0, :, :]
                    fake_frames = fake_videos[:, :, 0, :, :]
                    if fid_conf is None:
                        fid_score = compute_fid(real_frames, fake_frames, device=device)
                    lpips_score = compute_lpips(real_frames, fake_frames, device=device)
                    psnr_score = compute_psnr(real_frames, fake_frames)
                    ssim_score = compute_ssim(real_frames, fake_frames)
                    
                    fvd_scores.append(fvd_score)
                    if fid_conf is None:
                        fid_scores.append(fid_score)
                    lpips_scores.append(lpips_score)
                    psnr_scores.append(psnr_score)
                    ssim_scores.append(ssim_score)
            
            # Average metrics
            avg_fvd = sum(fvd_scores) / len(fvd_scores)
            avg_fid = sum(fid_scores) / len(fid_scores) if fid_scores else float("inf")
            avg_lpips = sum(lpips_scores) / len(lpips_scores)
            avg_psnr = sum(psnr_scores) / len(psnr_scores)
            avg_ssim = sum(ssim_scores) / len(ssim_scores)

            fid_score = None
            if fid_conf is not None:
                n_samples = int(fid_conf.get('n_samples', num_eval_batches * config['batch_size']))
                n_frames = fid_conf.get('n_frames', None)
                stat_file = fid_conf.get('stat_file', None)
                feat_batch = int(fid_conf.get('batchsize', config['batch_size']))
                real_features_list = []
                fake_features_list = []
                collected = 0
                with torch.no_grad():
                    pbar = tqdm(test_loader, desc="FVD/FID Evaluation", leave=False, ncols=80)
                    for batch in pbar:
                        if collected >= n_samples:
                            break
                        real_videos = batch.to(device)
                        remaining = n_samples - collected
                        if real_videos.size(0) > remaining:
                            real_videos = real_videos[:remaining]
                        fake_videos = diffusion.sample(
                            (real_videos.size(0), config['channels'], config['time_steps'], config['image_size'], config['image_size'])
                        )
                        if stat_file is None:
                            real_feat = extract_features_from_videos(real_videos, device=device, batchsize=feat_batch, n_frames=n_frames)
                            real_features_list.append(real_feat)
                        fake_feat = extract_features_from_videos(fake_videos, device=device, batchsize=feat_batch, n_frames=n_frames)
                        fake_features_list.append(fake_feat)
                        collected += real_videos.size(0)
                fake_features = torch.cat(fake_features_list, dim=0)
                real_features = torch.cat(real_features_list, dim=0) if real_features_list else None
                fid_score = compute_fid_from_features(fake_features, real_features=real_features, stat_file=stat_file)
                avg_fid = fid_score

            is_mean = None
            is_std = None
            if is_conf is not None:
                is_mean, is_std = compute_inception_score_from_sampler(
                    sample_fn=lambda n: diffusion.sample(
                        (n, config['channels'], config['time_steps'], config['image_size'], config['image_size'])
                    ),
                    n_samples=int(is_conf.get('n_samples', 1000)),
                    batchsize=int(is_conf.get('batchsize', 32)),
                    n_frames=is_conf.get('n_frames', None),
                    splits=int(is_conf.get('splits', 10)),
                    device=device
                )
            
            # Update best metrics (double optimal: both FVD and FID should improve)
            is_best = avg_fvd < best_fvd and avg_fid < best_fid
            if is_best:
                best_fvd = avg_fvd
                best_fid = avg_fid
                best_epoch = epoch + 1
                # Save best weights
                best_checkpoint = os.path.join(
                    config['checkpoint_dir'],
                    f"best_ep-{best_epoch}_fvd-{best_fvd:.3f}_fid-{best_fid:.3f}.pth"
                )
                torch.save(model.state_dict(), best_checkpoint)
                print(f"New best model saved at epoch {best_epoch}")
            
            # Print metrics with best comparison
            print(f"Epoch {epoch+1}/{config['epochs']}:")
            print(f"  Average FVD: {avg_fvd:.6f} (Best: {best_fvd:.6f} at epoch {best_epoch})")
            print(f"  Average FID: {avg_fid:.6f} (Best: {best_fid:.6f} at epoch {best_epoch})")
            print(f"  Average LPIPS: {avg_lpips:.6f}")
            print(f"  Average PSNR: {avg_psnr:.6f}")
            print(f"  Average SSIM: {avg_ssim:.6f}")
            if is_mean is not None:
                print(f"  Inception Score: {is_mean:.6f} ± {is_std:.6f}")
            
            # Log metrics
            writer.add_scalar('Metrics/Average_FVD', avg_fvd, epoch+1)
            writer.add_scalar('Metrics/Average_FID', avg_fid, epoch+1)
            writer.add_scalar('Metrics/Best_FVD', best_fvd, epoch+1)
            writer.add_scalar('Metrics/Best_FID', best_fid, epoch+1)
            writer.add_scalar('Metrics/Average_LPIPS', avg_lpips, epoch+1)
            writer.add_scalar('Metrics/Average_PSNR', avg_psnr, epoch+1)
            writer.add_scalar('Metrics/Average_SSIM', avg_ssim, epoch+1)
            if is_mean is not None:
                writer.add_scalar('Metrics/Inception_Score_Mean', is_mean, epoch+1)
                writer.add_scalar('Metrics/Inception_Score_Std', is_std, epoch+1)
            
            # Switch back to train mode
            model.train()

    # Calculate total training time
    total_training_time = time.time() - total_start_time
    total_training_time_min = total_training_time / 60
    total_training_time_h = total_training_time / 3600
    
    print("\n--- Training Complete ---")
    print(f"Total training time: {total_training_time:.2f}s ({total_training_time_min:.2f}min, {total_training_time_h:.2f}h)")
    
    # Close TensorBoard writer
    writer.close()
    
    print("Done.")

if __name__ == "__main__":
    main()
