import torch

class GradCam:
    def __init__(self, pretained, target_layer):
        self.pretained = pretained
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0] if grad_out[0] is not None else torch.zeros_like(grad_in[0])
        
        def forward_hook(module, input, output):
            self.activations = output if output is not None else torch.zeros_like(input[0])
        
        for name, module in self.pretained.named_modules():
            if name == self.target_layer:
                module.register_forward_hook(forward_hook)
                module.register_backward_hook(backward_hook)
        
    def forward(self, input):
        return self.pretained(input)
    
    def get_activations_gradient(self):
        return self.gradients

    def get_activations(self):
        return self.activations