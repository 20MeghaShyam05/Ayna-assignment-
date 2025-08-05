import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os


def denormalize_image(tensor):
    """
    Denormalize image tensor from [-1, 1] to [0, 1] range.
    
    Args:
        tensor: Image tensor in [-1, 1] range
    
    Returns:
        Denormalized tensor in [0, 1] range
    """
    return (tensor + 1) / 2


def normalize_image(tensor):
    """
    Normalize image tensor from [0, 1] to [-1, 1] range.
    
    Args:
        tensor: Image tensor in [0, 1] range
    
    Returns:
        Normalized tensor in [-1, 1] range
    """
    return tensor * 2 - 1


def tensor_to_pil(tensor):
    """
    Convert tensor to PIL Image.
    
    Args:
        tensor: Image tensor of shape (C, H, W) in [0, 1] range
    
    Returns:
        PIL Image
    """
    if tensor.dim() == 4:  # Batch dimension
        tensor = tensor[0]
    
    # Ensure tensor is in [0, 1] range
    tensor = torch.clamp(tensor, 0, 1)
    
    # Convert to PIL
    transform = transforms.ToPILImage()
    return transform(tensor)


def pil_to_tensor(pil_image, normalize=True):
    """
    Convert PIL Image to tensor.
    
    Args:
        pil_image: PIL Image
        normalize: Whether to normalize to [-1, 1] range
    
    Returns:
        Image tensor
    """
    transform = transforms.ToTensor()
    tensor = transform(pil_image)
    
    if normalize:
        tensor = normalize_image(tensor)
    
    return tensor


def visualize_batch(inputs, predictions, targets, colors, save_path=None, title="Batch Visualization"):
    """
    Visualize a batch of inputs, predictions, and targets.
    
    Args:
        inputs: Input images tensor (B, C, H, W)
        predictions: Predicted images tensor (B, C, H, W)
        targets: Target images tensor (B, C, H, W)
        colors: List of color names
        save_path: Optional path to save the visualization
        title: Title for the plot
    """
    batch_size = min(inputs.size(0), 4)  # Show max 4 samples
    
    # Denormalize images
    inputs = denormalize_image(inputs)
    predictions = denormalize_image(predictions)
    targets = denormalize_image(targets)
    
    fig, axes = plt.subplots(3, batch_size, figsize=(4 * batch_size, 12))
    
    if batch_size == 1:
        axes = axes.reshape(-1, 1)
    
    for i in range(batch_size):
        # Input image
        axes[0, i].imshow(inputs[i].permute(1, 2, 0).cpu().numpy())
        axes[0, i].set_title(f'Input ({colors[i]})')
        axes[0, i].axis('off')
        
        # Predicted image
        axes[1, i].imshow(predictions[i].permute(1, 2, 0).cpu().numpy())
        axes[1, i].set_title('Predicted')
        axes[1, i].axis('off')
        
        # Target image
        axes[2, i].imshow(targets[i].permute(1, 2, 0).cpu().numpy())
        axes[2, i].set_title('Target')
        axes[2, i].axis('off')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def create_color_grid():
    """
    Create a visual grid showing all available colors.
    
    Returns:
        Dictionary mapping color names to RGB values
    """
    color_map = {
        'red': (255, 0, 0),
        'green': (0, 255, 0),
        'blue': (0, 0, 255),
        'yellow': (255, 255, 0),
        'cyan': (0, 255, 255),
        'magenta': (255, 0, 255),
        'orange': (255, 165, 0),
        'purple': (128, 0, 128)
    }
    
    # Create visualization
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.flatten()
    
    for i, (color_name, rgb) in enumerate(color_map.items()):
        color_patch = np.ones((100, 100, 3), dtype=np.uint8)
        color_patch[:, :] = rgb
        
        axes[i].imshow(color_patch)
        axes[i].set_title(color_name.capitalize())
        axes[i].axis('off')
    
    plt.suptitle('Available Colors', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    return color_map


def calculate_metrics(predictions, targets):
    """
    Calculate evaluation metrics.
    
    Args:
        predictions: Predicted images tensor (B, C, H, W)
        targets: Target images tensor (B, C, H, W)
    
    Returns:
        Dictionary of metrics
    """
    # Ensure tensors are in the same range
    predictions = torch.clamp(predictions, -1, 1)
    targets = torch.clamp(targets, -1, 1)
    
    # MSE Loss
    mse = torch.mean((predictions - targets) ** 2).item()
    
    # L1 Loss
    l1 = torch.mean(torch.abs(predictions - targets)).item()
    
    # PSNR (Peak Signal-to-Noise Ratio)
    psnr = 20 * torch.log10(2.0 / torch.sqrt(torch.mean((predictions - targets) ** 2)))
    psnr = psnr.item()
    
    return {
        'mse': mse,
        'l1': l1,
        'psnr': psnr
    }


def load_and_preprocess_image(image_path, image_size=256):
    """
    Load and preprocess an image for inference.
    
    Args:
        image_path: Path to the image file
        image_size: Size to resize the image to
    
    Returns:
        Preprocessed image tensor
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
    ])
    
    # Apply transforms and add batch dimension
    image_tensor = transform(image).unsqueeze(0)
    
    return image_tensor


def create_color_condition(color_name, color_to_idx):
    """
    Create one-hot encoded color condition.
    
    Args:
        color_name: Name of the color
        color_to_idx: Dictionary mapping color names to indices
    
    Returns:
        One-hot encoded color tensor
    """
    num_colors = len(color_to_idx)
    color_condition = torch.zeros(1, num_colors)
    
    if color_name in color_to_idx:
        color_idx = color_to_idx[color_name]
        color_condition[0, color_idx] = 1.0
    else:
        raise ValueError(f"Unknown color: {color_name}. Available colors: {list(color_to_idx.keys())}")
    
    return color_condition


def save_model_for_inference(model, color_info, save_path):
    """
    Save model and metadata for easy inference loading.
    
    Args:
        model: Trained model
        color_info: Color information dictionary
        save_path: Path to save the model
    """
    torch.save({
        'model_state_dict': model.state_dict(),
        'color_info': color_info,
        'model_config': {
            'num_colors': color_info['num_colors'],
            'n_channels': 3,
            'n_classes': 3,
        }
    }, save_path)
    
    print(f"Model saved to {save_path}")


def load_model_for_inference(model_class, save_path, device='cpu'):
    """
    Load model for inference.
    
    Args:
        model_class: Model class to instantiate
        save_path: Path to the saved model
        device: Device to load the model on
    
    Returns:
        Loaded model and color info
    """
    checkpoint = torch.load(save_path, map_location=device)
    
    # Extract model configuration
    model_config = checkpoint['model_config']
    color_info = checkpoint['color_info']
    
    # Initialize model
    model = model_class(**model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, color_info


def compare_models(model1, model2, test_loader, device='cpu'):
    """
    Compare two models on a test dataset.
    
    Args:
        model1: First model
        model2: Second model
        test_loader: Test data loader
        device: Device to run inference on
    
    Returns:
        Comparison results
    """
    model1.eval()
    model2.eval()
    
    results = {'model1': [], 'model2': []}
    
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch['input_image'].to(device)
            colors = batch['color_onehot'].to(device)
            targets = batch['output_image'].to(device)
            
            # Model 1 predictions
            pred1 = model1(inputs, colors)
            metrics1 = calculate_metrics(pred1, targets)
            results['model1'].append(metrics1)
            
            # Model 2 predictions
            pred2 = model2(inputs, colors)
            metrics2 = calculate_metrics(pred2, targets)
            results['model2'].append(metrics2)
    
    # Average metrics
    for model_name in results:
        avg_metrics = {}
        for metric in results[model_name][0].keys():
            avg_metrics[metric] = np.mean([r[metric] for r in results[model_name]])
        results[model_name] = avg_metrics
    
    return results


def create_inference_examples():
    """
    Create example input-output pairs for the inference notebook.
    """
    examples = [
        {'polygon': 'triangle.png', 'color': 'red', 'description': 'Red triangle'},
        {'polygon': 'square.png', 'color': 'blue', 'description': 'Blue square'},
        {'polygon': 'hexagon.png', 'color': 'green', 'description': 'Green hexagon'},
        {'polygon': 'circle.png', 'color': 'yellow', 'description': 'Yellow circle'},
        {'polygon': 'star.png', 'color': 'purple', 'description': 'Purple star'},
    ]
    
    return examples