import torch
import clip
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from io import BytesIO
import requests

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def extract_clip_image_features(image_url: str):
    response = requests.get(image_url)
    if response.status_code != 200:
        raise ValueError("Failed to retrieve image from the provided URL.")
    image_tensor = preprocess(Image.open(BytesIO(response.content))).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image_tensor)
    return image_features[0].cpu().tolist()

def extract_clip_text_features(text: str):
    if not isinstance(text, str):
        raise ValueError("Input text must be a string.")
    text_tokens = clip.tokenize([text]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
    return text_features[0].cpu().tolist()

def extract_image_and_text_features(image_url: str, text: str, image_weight=0.6, text_weight=0.4):
    query_image_features = extract_clip_image_features(image_url)
    query_text_features = extract_clip_text_features(text)

    query_image_features = np.array(query_image_features)
    query_text_features = np.array(query_text_features)

    combined_features = image_weight * query_image_features + text_weight * query_text_features

    return combined_features.tolist()