import torch
import torch.nn as nn
import torch.optim as optim
import cv2
from PIL import Image
import clip
import numpy as np
from skimage import io, transform
import torchvision.transforms as transforms
from glob import glob
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True



classes = ('document','else','face','sexual')

DATA_PATH_TRAINING_LIST = glob('./dataset/train/*/*.jpg')
DATA_PATH_TESTING_LIST = glob('./dataset/test/*/*.jpg')
BATCH_SIZE=32
EPOCH= 10

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4452, 0.4457, 0.4464), (0.2592, 0.2596, 0.2600)),
])
def get_label(data_path_list):
    label_list = []
    for path in data_path_list:
        # 뒤에서 두번째가 class다.
        label_list.append(path.split('/')[-2])
    return label_list

class MyClipDataset(Dataset):
    def __init__(self, data_path_list, classes, transform=None):
        self.path_list = data_path_list
        self.label = get_label(data_path_list)
        self.transform = transform
        self.classes = classes

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        #image = io.imread(self.path_list[idx])
        #image = cv2.imread(self.path_list[idx]) 
        image = Image.open(self.path_list[idx]).convert('RGB')

        if self.transform:
            image = self.transform(image)
        return image, self.label[idx]


train_dataloader = torch.utils.data.DataLoader(
    MyClipDataset(
        DATA_PATH_TRAINING_LIST, 
        classes,
        transform=transform
    ),
    batch_size=32,
    shuffle = True
)

test_dataloader = torch.utils.data.DataLoader(
    MyClipDataset(
        DATA_PATH_TESTING_LIST, 
        classes,
        transform=transform
    ),
    batch_size=32,
    shuffle = False
)
#https://github.com/openai/CLIP/issues/57
def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float() 
        p.grad.data = p.grad.data.float() 

device = "cuda:0" if torch.cuda.is_available() else "cpu" # If using GPU then use mixed precision training.
model, preprocess = clip.load("ViT-B/32",device=device,jit=False) #Must set jit=False for training
if device == "cpu":
  model.float()
else :
  clip.model.convert_weights(model) # Actually this line is unnecessary since clip by default already on float16

loss_img = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=5e-5,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2) #Params used from paper, the lr is smaller, more safe for fine tuning to new dataset

for epoch in range(EPOCH):
  for batch in train_dataloader:
      optimizer.zero_grad()
      print("?")
      list_image,list_txt = batch #list_images is list of image in numpy array(np.uint8), or list of PIL images
      images= torch.stack([preprocess(Image.fromarray((img.permute(1, 2, 0).numpy() * 255).astype(np.uint8))) for img in list_image],dim=0).to(device) # omit the Image.fromarray if the images already in PIL format, change this line to images=list_image if using preprocess inside the dataset class

      texts = clip.tokenize(list_txt).to(device)
    
      logits_per_image, logits_per_text = model(images, texts)

      ground_truth = torch.arange(BATCH_SIZE,dtype=torch.long,device=device)

      total_loss = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2
      total_loss.backward()

      if device == "cpu":
         optimizer.step()
      else : 
        convert_models_to_fp32(model)
        optimizer.step()
        clip.model.convert_weights(model)

print('Finished Training')
torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': total_loss,
        }, f"model/model_10.pt") #just change to your preferred folder/filename

print('Saved trained model')
