import torch
import torch.nn as nn
import torch.optim as optim
#load image
from PIL import Image
#convert to tensor
import torchvision.transforms as transforms
#load vgg19
import torchvision.models as models
from torchvision.utils import save_image

#get all conv layers
model = models.vgg19(pretrained=True).features

#take conv11,21,31,41,51. Number changes after maxpool

class VGG(nn.Module):
    def __init__(self):
        super(VGG,self).__init__()
        
        self.chosen_features= ['0','5','10','19','28']
        #layers of the model
        self.model=models.vgg19(pretrained=True).features[:29]

    def forward(self,x):
        features=[]

        for layer_num,layer in enumerate(self.model):
            x=layer(x)
            if str(layer_num) in self.chosen_features:
                features.append(x)
        
        return features

def load_image(image_name):
    image=Image.open(image_name)
    #add additional dimension for batch size
    image=loader(image).unsqueeze(0)
    return image.to(device)

device=torch.device('cuda' if torch.cuda.is_available else 'cpu')
#image_size=1024

loader=transforms.Compose([
    #transforms.Resize((image_size,image_size))
    transforms.ToTensor(),
])

original_img=load_image("C:/Users/NavneethBhat/Desktop/Code/Implementations/Pics/nav_ise_new.jpeg")
style_img=load_image("C:/Users/NavneethBhat/Desktop/Code/Implementations/Pics/goldie.png")

generated=original_img.clone().requires_grad_(True)


#Hyperparameters
total_steps=6000
learning_rate=0.01
alpha= 1
beta=0.01
optimizer=optim.Adam([generated],lr=learning_rate)
model=VGG().to(device).eval()

for step in range(total_steps):
    generated_features=model(generated)
    original_img_features=model(original_img)
    style_features=model(style_img)

    style_loss=original_loss=0

    for gen_feature,orig_feature,style_feature in zip(
        generated_features,original_img_features,style_features
    ):
        batch_size,channel,height,width=gen_feature.shape
        original_loss+=torch.mean((gen_feature-orig_feature)**2)

        #compute gram matrix mm=>matrix multiply
        G=gen_feature.view(channel,height*width).mm(gen_feature.view(channel,height*width).t())
        A=style_feature.view(channel,height*width).mm(style_feature.view(channel,height*width).t())
        style_loss+=torch.mean((G-A)**2)

    total_loss=alpha*original_loss+beta*style_loss
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if step%100==0:
        print(total_loss)
        save_image(generated,'gen.png')


