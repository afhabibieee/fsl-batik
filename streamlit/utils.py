import torch
import cv2
from gradcam import GradCam


def class_activation_map(embedding, query_image_tensor):
    # Load model
    pretrained = torch.nn.Sequential(*(list(embedding.backbone.children())[:-2]))
    pretrained.eval()

    # Set the target layer (last layer before avg/global pool or layer you're interested in)
    target_layer = [name for name, _ in pretrained.named_modules()][-1]

    # Initialize GradCam
    grad_cam = GradCam(pretrained, target_layer)

    # Forward pass
    output = grad_cam.forward(query_image_tensor)

    # Backward pass
    output.backward(torch.ones_like(output))

    # Get gradients and activations
    gradients = grad_cam.get_activations_gradient()
    activations = grad_cam.get_activations()

    # Compute the weights for the final linear combination
    alpha_k = torch.mean(gradients, dim=(2, 3), keepdim=True)

    # Perform the linear combination to compute the CAM
    cam = torch.sum(alpha_k * activations, dim=1, keepdim=True)

    # ReLU on the CAM (only keep the positive values)
    cam = torch.nn.functional.relu(cam)

    # Normalize the CAM to be between 0 and 1
    cam -= cam.min()
    cam /= cam.max()

    # Detach and move the CAM to CPU
    cam = cam.detach().cpu().numpy()

    # Resize the cam to the size of the query image
    cam = cv2.resize(cam[0, 0, :, :], (query_image_tensor.shape[2], query_image_tensor.shape[3]))

    return cam