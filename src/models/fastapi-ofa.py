from fastapi import FastAPI, File, Form, UploadFile
import uvicorn
from pyngrok import ngrok
import nest_asyncio
from ofatest import *  # import your wrapper

# Fix for running Uvicorn inside interactive environments
nest_asyncio.apply()

# Create your FastAPI app
app = FastAPI()

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    task: str = Form(...)
):
    # Read file into memory
    file_bytes = await file.read()
    image_buffer = BytesIO(file_bytes)

    # Call OFA model directly with in-memory image
    description = ofa_model(image_buffer, prompt=task)

    return {"description": description}

@app.get("/")
def home():
    return {"message": "Hello from FastAPI running with ngrok!"}

if __name__ == "__main__":
    # Start an ngrok tunnel to the FastAPI app
    public_url = ngrok.connect(8000)
    print(f"Public URL: {public_url}")

    # Start the FastAPI server
    uvicorn.run(app, host="0.0.0.0", port=8000)
