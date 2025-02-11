import pandas as pd
import numpy as np
import requests
import torch
from PIL import Image
from io import BytesIO
from torchvision import models, transforms
from sentence_transformers import SentenceTransformer

# Load dataset
file_path = r"..\data\VICR_Sample250.csv"  # Update path if needed
df = pd.read_csv(file_path)

# Compute rounded average rating
rating_columns = ["ratings", "rating_2", "rating_3", "rating_4", "rating_5", "rating_6", "rating_7"]
df["Rating"] = df[rating_columns].mean(axis=1).round().astype(int)

# Drop rows with missing images or captions before processing
df.dropna(subset=["image", "caption"], inplace=True)

# Load ResNet-50 for image embeddings (Pretrained on ImageNet)
device = "cpu"  # Using CPU only
resnet_model = models.resnet50(pretrained=True)
resnet_model = resnet_model.eval()  # Set model to evaluation mode
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load Sentence Transformers for text embeddings
text_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Image preprocessing
def preprocess_image(image_url):
    try:
        response = requests.get(image_url, timeout=10)
        image = Image.open(BytesIO(response.content)).convert("RGB")
        return image
    except Exception as e:
        print(f"Error downloading {image_url}: {e}")
        return None

# Generate image embedding using ResNet-50
def get_image_embedding(image):
    if image is None:
        return None  # Return None instead of NaN
    image_tensor = image_transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        embedding = resnet_model(image_tensor).numpy().flatten()
    return embedding

# Generate text embedding using Sentence Transformers
def get_text_embedding(caption):
    if not isinstance(caption, str):
        return None  # Return None instead of NaN
    with torch.no_grad():
        embedding = text_model.encode(caption)
    return embedding

# Process dataset
image_embeddings = []
caption_embeddings = []

for index, row in df.iterrows():
    image = preprocess_image(row["image"])
    img_embedding = get_image_embedding(image)
    txt_embedding = get_text_embedding(row["caption"])
    
    image_embeddings.append(img_embedding)
    caption_embeddings.append(txt_embedding)

# Add embeddings to DataFrame
df["image_embedding"] = image_embeddings
df["caption_embedding"] = caption_embeddings

# Drop rows with missing embeddings
df_cleaned = df.dropna(subset=["image_embedding", "caption_embedding"])

# Convert embeddings to lists for CSV storage
df_cleaned["image_embedding"] = df_cleaned["image_embedding"].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
df_cleaned["caption_embedding"] = df_cleaned["caption_embedding"].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)

# Concatenate embeddings safely
df_cleaned["Concatnated_image_caption"] = df_cleaned.apply(
    lambda row: np.concatenate([row["image_embedding"], row["caption_embedding"]]).tolist(), axis=1
)

# Select final columns and save
final_df = df_cleaned[["image_embedding", "caption_embedding", "Concatnated_image_caption", "Rating"]]
final_csv_path = "VICR_Sample250_Cleaned.csv"
final_df.to_csv(final_csv_path, index=False)

print(f"Final dataset saved as {final_csv_path} with shape {final_df.shape}")
