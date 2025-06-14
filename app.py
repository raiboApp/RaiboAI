from fastapi import FastAPI
from pydantic import BaseModel
import clip_utils

app = FastAPI()

class TextQuery(BaseModel):
    text_query: str

class ImageQuery(BaseModel):
    image_url: str

class ImageTextQuery(BaseModel):
    image_url: str
    text_query: str

@app.post("/embed-text")
def search_text(query: TextQuery):
    text_features = clip_utils.extract_clip_text_features(query.text_query)
    if text_features is None:
        return {"error": "Failed to extract text features."}
    return {"text_features": text_features}

@app.post("/embed-image")
def search_image(query: ImageQuery):
    image_features = clip_utils.extract_clip_image_features(query.image_url)
    if image_features is None:
        return {"error": "Failed to extract image features."}
    return {"image_features": image_features}

@app.post("/embed-image-text")
def search_image_text(query: ImageTextQuery):
    print("hello")
    combined_features = clip_utils.extract_image_and_text_features(
        query.image_url, query.text_query
    )
    if combined_features is None:
        return {"error": "Failed to extract combined features."}
    return {"combined_features": combined_features}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=5000, reload=True)

