from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import clip_utils
import requests

app = FastAPI()

@app.post("/embed-text")
def search_text(text_query: str = Form(...)):
    text_embedding = clip_utils.extract_clip_text_features(text_query)
    return {"text_embedding": text_embedding}

@app.post("/embed-image")
async def search_image(image: UploadFile = File(None), image_url: str = Form(None)):
    try:
        if image:
            image_bytes = await image.read()
        elif image_url:
            response = requests.get(image_url)
            response.raise_for_status()
            image_bytes = response.content
        else:
            return JSONResponse(
                status_code=400,
                content={"error": "Either an image file or an image URL must be provided."}
            )
        image_embedding = clip_utils.extract_clip_image_features(image_bytes)
        return {"image_embedding": image_embedding}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.post("/embed-image-text")
async def search_image_text(image: UploadFile = File(None), image_url: str = Form(None), text_query: str = Form(...)):
    try:
        if image:
            image_bytes = await image.read()
        elif image_url:
            response = requests.get(image_url)
            response.raise_for_status()
            image_bytes = response.content
        else:
            return JSONResponse(
                status_code=400,
                content={"error": "Either an image file or an image URL must be provided."}
            )
        combined_embedding = clip_utils.extract_image_and_text_features(
            image_bytes, text_query
        )
        return {"combined_embedding": combined_embedding}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

# if __name__ == "__main__":
#     import uvicorn
#     port = int(os.getenv("PORT", 8080))
#     uvicorn.run("app:app", host="0.0.0.0", port=port)

