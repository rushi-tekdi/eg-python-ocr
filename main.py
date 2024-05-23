# main.py
import os
import time
from datetime import datetime

import cv2
import pytesseract
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from pdf2image import convert_from_path
from PIL import Image
from starlette.middleware.cors import CORSMiddleware

app = FastAPI()

allowed_cors_origins = os.getenv("ALLOWED_CORS_ORIGINS", "http://localhost:4000")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4000", allowed_cors_origins],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/ocr-extract-text/")
async def upload_file(file: UploadFile = File(...)):
    uploaded_file_path = ""
    image_path = ""
    try:
        # Get the file extension
        extension = os.path.splitext(file.filename)[1]

        # Generate a unique file name
        uploaded_file_path = f"uploads/{file.filename.split('.')[0]}_{int(time.mktime(datetime.now().timetuple()))}{extension}"

        # Save the file to disk
        with open(uploaded_file_path, "wb") as buffer:
            buffer.write(file.file.read())

        # # Convert the PDF to an image
        images = convert_from_path(uploaded_file_path)
        print(images)

        # # Save the image
        image_path = f"{os.path.splitext(uploaded_file_path)[0]}.jpg"
        images[0].save(image_path, "JPEG")

        # Clean up the temporary files
        await delete_file(uploaded_file_path)

        text = await extract_text(image_path)

        # Delete file post success/error
        await delete_file(image_path)

        return {
            "success": True,
            "message": "Data extracted successfully!",
            "data": text,
        }
    except Exception as exception:
        # Delete file post success/error
        await delete_file(uploaded_file_path)
        await delete_file(image_path)

        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "message": "Error in extracting data: " + repr(exception),
            },
        )


async def extract_text(image_path):
    processed_image_path = ""
    try:
        processed_image_path = await preprocess_image(image_path)

        # Use Tesseract to extract text
        text = pytesseract.image_to_string(Image.open(processed_image_path))

        # Delete file post success/error
        await delete_file(processed_image_path)

        return text
    except Exception as exception:
        # Delete file post success/error
        await delete_file(processed_image_path)
        print(repr(exception))


async def preprocess_image(image_path):
    # Read image using OpenCV
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # Save the processed image temporarily
    processed_image_path = (
        f"uploads/processed_image_{int(time.mktime(datetime.now().timetuple()))}.png"
    )
    cv2.imwrite(processed_image_path, gray)

    return processed_image_path


async def delete_file(file_path):
    # Delete processed image once text is extracted
    # Check if the file exists
    if os.path.exists(file_path):
        # Delete the file
        os.remove(file_path)

    return True
