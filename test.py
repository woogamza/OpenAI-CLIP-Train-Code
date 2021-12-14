import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
# Must set jit=False for training
model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
checkpoint = torch.load("model/model_10.pt")

# Use these 3 lines if you use default model setting(not training setting) of the clip. For example, if you set context_length to 100 since your string is very long during training, then assign 100 to checkpoint['model_state_dict']["context_length"]

model.load_state_dict(checkpoint['model_state_dict'])

image = preprocess(Image.open(
    "../다운로드.jpg")).unsqueeze(0).to(device)
text = clip.tokenize(["document", "dog", "face", "sexual"]).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)

    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]
