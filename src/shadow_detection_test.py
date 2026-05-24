import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import random
from model import UNet



def test_on_test_set(model_path='models/shadow_detection.pth', 
                     test_image_folder='data/ISTD_Dataset/test/test_A',
                     test_mask_folder='data/ISTD_Dataset/test/test_B',
                     num_samples=5,
                     random_seed=None):
    
    if random_seed is not None:
        random.seed(random_seed)
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {model_path}...")
    model = UNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Model loaded successfully!")
    
    # Get all test images
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    test_images = [f for f in os.listdir(test_image_folder) 
                   if any(f.lower().endswith(ext) for ext in image_extensions)]
    test_images.sort()
    
    print(f"\nFound {len(test_images)} test images")
    
    # RANDOMLY SELECT images
    if num_samples and num_samples < len(test_images):
        test_images = random.sample(test_images, num_samples)
        print(f"Randomly selected {num_samples} images for testing")
    
    # Create output directory
    os.makedirs('outputs/test_results', exist_ok=True)
    
    # Store metrics
    iou_scores = []
    accuracy_scores = []
    
    # Test each randomly selected image
    for idx, img_file in enumerate(test_images[:num_samples] if num_samples else test_images):
        print(f"[{idx+1}/{len(test_images[:num_samples] if num_samples else test_images)}] Testing: {img_file}")
        
        # Load test image
        img_path = os.path.join(test_image_folder, img_file)
        image = Image.open(img_path).convert("RGB").resize((256, 256))
        image_np = np.array(image, dtype=np.float32) / 255.0
        
        # Load ground truth mask
        mask_path = os.path.join(test_mask_folder, img_file)
        if not os.path.exists(mask_path):
            print(f"    No ground truth mask found, skipping...")
            continue
            
        gt_mask = Image.open(mask_path).convert("L").resize((256, 256))
        gt_mask_np = np.array(gt_mask, dtype=np.float32) / 255.0
        gt_binary = (gt_mask_np > 0.5).astype(np.float32)
        
        # Predict shadow mask
        image_tensor = torch.tensor(image_np).permute(2, 0, 1).unsqueeze(0).to(device)
        
        with torch.no_grad():
            shadow_mask = model(image_tensor)
        
        shadow_mask = shadow_mask.squeeze().cpu().numpy()
        pred_binary = (shadow_mask > 0.5).astype(np.float32)
        
        # Calculate metrics
        intersection = np.logical_and(pred_binary, gt_binary).sum()
        union = np.logical_or(pred_binary, gt_binary).sum()
        iou = intersection / union if union > 0 else 0
        accuracy = (pred_binary == gt_binary).mean()
        
        iou_scores.append(iou)
        accuracy_scores.append(accuracy)
        
        print(f"   IoU: {iou:.4f} | Accuracy: {accuracy:.4f}")
        
        # Save visualization - ONLY 3 COLUMNS
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Column 1: Original Image
        axes[0].imshow(image_np)
        axes[0].set_title('Original Image (with shadows)', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # Column 2: Your Prediction
        axes[1].imshow(pred_binary, cmap='gray', vmin=0, vmax=1)
        axes[1].set_title(f'Your Prediction (IoU: {iou:.3f})', fontsize=12, fontweight='bold')
        axes[1].axis('off')
        
        # Column 3: Ground Truth
        axes[2].imshow(gt_binary, cmap='gray', vmin=0, vmax=1)
        axes[2].set_title('Ground Truth (Correct Answer)', fontsize=12, fontweight='bold')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'outputs/test_results/{img_file}_result.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   Saved to: outputs/test_results/{img_file}_result.png")
        print()
    
    # Print summary (ONLY your original metrics)
    print("="*50)
    print("SUMMARY")
    print("="*50)
    
    if iou_scores:
        print(f"Average IoU:        {np.mean(iou_scores):.4f}")
        print(f"Average Accuracy:   {np.mean(accuracy_scores):.4f}")
        print(f"Best IoU:           {np.max(iou_scores):.4f}")
        print(f"Worst IoU:          {np.min(iou_scores):.4f}")
    
    print(f"\n Test complete! Results saved to outputs/test_results/")
    
    # Create summary plot (keeping your original function)
    if iou_scores:
        create_summary_plot(iou_scores, accuracy_scores, test_images[:num_samples] if num_samples else test_images)

def create_summary_plot(iou_scores, accuracy_scores, image_names):
    """Create a summary bar chart of test results"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # IoU bar chart
    x = range(len(iou_scores))
    axes[0].bar(x, iou_scores, color='skyblue', edgecolor='black')
    axes[0].set_xlabel('Test Image')
    axes[0].set_ylabel('IoU Score')
    axes[0].set_title('Intersection over Union (IoU) - Higher is better', fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f'{i+1}' for i in x], rotation=45)
    axes[0].set_ylim([0, 1])
    axes[0].axhline(y=np.mean(iou_scores), color='red', linestyle='--', label=f'Mean: {np.mean(iou_scores):.4f}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy bar chart
    axes[1].bar(x, accuracy_scores, color='lightgreen', edgecolor='black')
    axes[1].set_xlabel('Test Image')
    axes[1].set_ylabel('Accuracy Score')
    axes[1].set_title('Accuracy - Higher is better', fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f'{i+1}' for i in x], rotation=45)
    axes[1].set_ylim([0, 1])
    axes[1].axhline(y=np.mean(accuracy_scores), color='red', linestyle='--', label=f'Mean: {np.mean(accuracy_scores):.4f}')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outputs/test_results/summary_plot.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f" Summary plot saved to outputs/test_results/summary_plot.png")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test shadow detection on official test set')
    parser.add_argument('--num', type=int, default=10,
                        help='Number of test images to process (default: 10)')
    parser.add_argument('--all', action='store_true',
                        help='Test on all test images')
    parser.add_argument('--model', type=str, default='models/shadow_detection.pth',
                        help='Path to trained model')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducible results')
    
    args = parser.parse_args()
    
    num_samples = None if args.all else args.num
    test_on_test_set(
        model_path=args.model,
        test_image_folder='data/ISTD_Dataset/test/test_A',
        test_mask_folder='data/ISTD_Dataset/test/test_B',
        num_samples=num_samples,
        random_seed=args.seed
    )