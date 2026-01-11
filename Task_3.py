import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.utils import save_image

# 1. Configuration & Device Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
imsize = 512 if torch.cuda.is_available() else 128  # Adjust based on memory

loader = transforms.Compose([
    transforms.Resize((imsize, imsize)),
    transforms.ToTensor()])

def load_image(image_name):
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

# 2. Load Content and Style Images
# Replace these with your own file paths
content_img = load_image("content.jpg")
style_img = load_image("style.jpg")
input_img = content_img.clone() # Start with the content image

# 3. Import Pre-trained VGG19
vgg = models.vgg19(pretrained=True).features.to(device).eval()

# Helper: Gram Matrix for Style
def get_gram_matrix(tensor):
    _, d, h, w = tensor.size()
    features = tensor.view(d, h * w)
    gram = torch.mm(features, features.t())
    return gram.div(d * h * w)

# 4. Optimization Loop
optimizer = optim.LBFGS([input_img.requires_grad_()])
run = [0]

while run[0] <= 300:
    def closure():
        input_img.data.clamp_(0, 1)
        optimizer.zero_grad()
        
        # In a real script, you'd extract features from specific 
        # layers here (e.g., Conv4_2 for content, Conv1_1...5_1 for style)
        # and calculate the MSE loss against the targets.
        
        loss = torch.mean((input_img - content_img)**2) # Simplified example
        loss.backward()
        
        if run[0] % 50 == 0:
            print(f"Step {run[0]}: Loss {loss.item()}")
        run[0] += 1
        return loss
    
    optimizer.step(closure)

# 5. Save Final Result
save_image(input_img, "stylized_output.png")
print("Task Complete: Image saved as stylized_output.png")