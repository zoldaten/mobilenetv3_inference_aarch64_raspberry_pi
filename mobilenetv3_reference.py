def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
#https://huggingface.co/docs/timm/models/mobilenet-v3

import urllib
import torch
from torchvision import transforms
import torchvision.models as models
from PIL import Image
from datetime import datetime

# 
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    # ImageNet weight
    weight = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
    model = models.mobilenet_v3_small(weight).to(DEVICE)
    model.eval()  
    
    
    input_image = Image.open('1.jpg').convert('RGB')
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0).to(DEVICE) # 
    
    #inputs = image_processor(image, return_tensors="pt")
    with torch.no_grad():        
            output = model(input_batch)
            
            #https://deeplearning.cms.waikato.ac.nz/user-guide/class-maps/IMAGENET/
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            #print(probabilities.shape)
            #To get the top-5 predictions class names:
            url, filename = ("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt", "imagenet_classes.txt")
            urllib.request.urlretrieve(url, filename)
            with open("imagenet_classes.txt", "r") as f:
                categories = [s.strip() for s in f.readlines()]
            # Print top categories per image
            top5_prob, top5_catid = torch.topk(probabilities, 5)
            for i in range(top5_prob.size(0)):
                print(categories[top5_catid[i]], top5_prob[i].item())
            # prints class names and probabilities like:
            # [('Samoyed', 0.6425196528434753), ('Pomeranian', 0.04062102362513542), ('keeshond', 0.03186424449086189), ('white wolf', 0.01739676296710968), ('Eskimo dog', 0.011717947199940681)]
        
if __name__ == '__main__':
    start = datetime.now()
    main()
    print(datetime.now()- start)
