# main.py
import os
import time
from datetime import datetime

import cv2
import pytesseract
# from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from pdf2image import convert_from_path
from PIL import Image
from starlette.middleware.cors import CORSMiddleware

#
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from pdf2image import convert_from_bytes
import pytesseract
from PIL import Image 
import io
import logging
import re
import tabula
import pandas as pd
from io import BytesIO
import numpy as np
from img2table.document import Image as LibImage
from img2table.ocr import TesseractOCR

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

allowed_cors_origins = os.getenv("ALLOWED_CORS_ORIGINS", "http://localhost:4000")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4000", allowed_cors_origins],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# below api uses tabula only for text pdf

def pdf_to_high_res_images(pdf_bytes, dpi=300):
    # Convert PDF to images with the specified DPI
    images = convert_from_bytes(pdf_bytes, dpi=dpi)
    return images

@app.post("/extract/")
async def extract_data(page_number: int = Form(...), file: UploadFile = File(...)):
    try:
        # Ensure the JAVA_HOME is set
        if not os.getenv('JAVA_HOME'):
            raise EnvironmentError("JAVA_HOME environment variable is not set.")

        pdf_data = await file.read()
        
        # print("pdf_data received ",pdf_data)
        # Convert PDF to high-resolution images
        images = pdf_to_high_res_images(pdf_data, dpi=100)

        if page_number < 0 or page_number >= len(images):
            raise HTTPException(status_code=400, detail="Page number out of range.")
        
        print("page length ",len(images))
        print("images",images)
        # Select the specified page image
        page_image = images[page_number]
        print("page_image",page_image)
        
        # Perform OCR on the image
        ocr_result = pytesseract.image_to_string(page_image)
        print("ocr_result",ocr_result)

        # Optionally, you can use Tesseract to directly extract tables in TSV format
        ocr_result_tsv = pytesseract.image_to_data(page_image, output_type=pytesseract.Output.DATAFRAME)
        
        # Find rows that likely belong to a table based on confidence level or other criteria
        table_data = ocr_result_tsv[ocr_result_tsv['conf'] > 60]
        
        if table_data.empty:
            raise HTTPException(status_code=400, detail="No table data found on the specified page.")
        
        # tabula pdf file
        pdf_path = BytesIO(pdf_data)
        print("pdf_path",pdf_path)

        # Use Tabula to extract tables from the specified page
        tables = tabula.read_pdf(pdf_path, pages=page_number+1, multiple_tables=True)
        print("tables f",tables)

        if not tables:
            raise HTTPException(status_code=400, detail="No tables found on the specified page.") 

        # Convert extracted tables to JSON format
        table_data = [table.to_dict(orient="records") for table in tables]

        return {"ocr_result": ocr_result, "table_data": table_data}

    except Exception as e:
        #logger.error(f"Error processing file: {e}")
        print("error ",e)
        return JSONResponse(status_code=500, content={"error": "Internal Server Error"})

# end below api uses tabula only for text pdf

# below api uses opencv only for image pdf
def preprocess_image_2(image):

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # save gray
    # cv2.imwrite("1_gray.jpg", gray)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # save gray
    # cv2.imwrite("2_blurred.jpg", blurred)
    
    # Apply adaptive thresholding to create a binary image
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    # save gray
    # cv2.imwrite("3_binary.jpg", binary)
    
    # Use morphological operations to clean up noise
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Dilate to connect text parts
    cleaned = cv2.dilate(cleaned, kernel, iterations=1)
    # save gray
    # cv2.imwrite("4_cleaned.jpg", cleaned)
    
    return cleaned

def preprocess_image_1(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY_INV)
    return binary

def detect_table_structure(binary_image):
    # Detect horizontal and vertical lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))

    horizontal_lines = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, horizontal_kernel)
    vertical_lines = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, vertical_kernel)

    # Combine the detected lines
    table_structure = cv2.add(horizontal_lines, vertical_lines)

    return table_structure

def extract_cells(image, table_structure):
    contours, _ = cv2.findContours(table_structure, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    cells = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 20 and h > 20:  # Filter out small contours that are unlikely to be table cells
            cell_image = image[y:y+h, x:x+w]
            cells.append((x, y, w, h, cell_image))
    
    # Sort cells by their position (top to bottom, left to right)
    cells = sorted(cells, key=lambda b: (b[1], b[0]))
    
    return cells

def extract_table_data_from_image(image_path):
    image = cv2.imread(image_path)
    # save
    cv2.imwrite("1_image.jpg", image)
    binary_image = preprocess_image_1(image)
    # save
    cv2.imwrite("2_binary_image.jpg", binary_image)

    table_structure = detect_table_structure(binary_image)
    # save
    cv2.imwrite("3_table_structure.jpg", table_structure)
    cells = extract_cells(image, table_structure)
    
    table_data = []
    for x, y, w, h, cell_image in cells:
        # Apply preprocessing specific to each cell to improve OCR results
        cell_gray = cv2.cvtColor(cell_image, cv2.COLOR_BGR2GRAY)
        cell_thresh = cv2.adaptiveThreshold(cell_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        cell_text = pytesseract.image_to_string(cell_thresh, config='--psm 6').strip()
        table_data.append(cell_text)
    
    return table_data

@app.post("/extract_table_data/")
async def process_pdf(file: UploadFile = File(...)):
    if file.content_type == 'application/pdf':
        try:
            # Convert PDF to image
            pdf_bytes = await file.read()
            images = convert_from_bytes(pdf_bytes)
            image = images[0]  # Assuming there is only one page in the PDF
            
            # Save the image temporarily
            image_path = 'temp_image.jpg'
            image.save(image_path, 'JPEG')


            #image2table

            # Definition of image from path
            img_from_path = LibImage(src=image_path)

            # Extract tables
            extracted_tables = img_from_path.extract_tables()
            # Display extracted tables
            table_img = cv2.imread(image_path)

            for table in extracted_tables:
                for row in table.content.values():
                    for cell in row:
                        cv2.rectangle(table_img, (cell.bbox.x1, cell.bbox.y1), (cell.bbox.x2, cell.bbox.y2), (255, 0, 0), 2)
                        
            print("table_img",table_img)
            test_table_img=Image.fromarray(table_img)
            cv2.imwrite("test.jpg", table_img)



            # Instantiation of the image
            # img = Image(src=image_path)

            # # Table identification
            # img_tables = img.extract_tables(ocr=ocr,
            #                           implicit_rows=False,
            #                           borderless_tables=False,
            #                           min_confidence=50)
            # for id_row, row in enumerate(table.content.values()):
            #     for id_col, cell in enumerate(row):
            #         x1 = cell.bbox.x1
            #         y1 = cell.bbox.y1
            #         x2 = cell.bbox.x2
            #         y2 = cell.bbox.y2
            #         value = cell.value
            # Result of table identification
            # print("img_tables ",img_tables)

            """[ExtractedTable(title=None, bbox=(10, 8, 745, 314),shape=(6, 3)),
            ExtractedTable(title=None, bbox=(936, 9, 1129, 111),shape=(2, 2))]"""

            # Extract table data from the image
            # table_data = extract_table_data_from_image(image_path)

            # Remove the temporary image file
            os.remove(image_path)

            # Return the extracted table data
            return {"table_data": []}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    else:
        raise HTTPException(status_code=415, detail="Unsupported Media Type")
    
# end below api uses opencv only for image pdf

@app.post("/ocr-extract-text/")
async def upload_file(file: UploadFile = File(...)):
    """
    Endpoint to upload a file and extract text from it using OCR.

    Args:
        file (UploadFile): The file to be uploaded.

    Returns:
        dict: A dictionary containing the success status, message, and extracted text.
    """
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

        """
        const pdf2picInstance = fromBuffer(pdfBuffer, {
				density: 500, // output pixels per inch
				saveFilename: 'page',
				savePath: outputDirectory, // save path for the images
				format: 'png', // image format
				width: 1500, // image width
				height: 2000, // image height
			});
        
        image width : 1500 height : 200 density :500 image from pdf and then read using opencv
        """

        # # Convert the PDF to an image
        # #images = convert_from_path(uploaded_file_path)
        images = convert_from_path(
            pdf_path=uploaded_file_path,
            fmt="jpeg",  # optional
            dpi=500,  # optional
            size=(1000, 1500),  # optional
        )

        # print(images)

        # # Save the image
        image_path = f"{os.path.splitext(uploaded_file_path)[0]}.jpeg"
        images[0].save(image_path, "JPEG")

        # Clean up the temporary files
        # await delete_file(uploaded_file_path)

        text = await extract_text(image_path)

        # Delete file post success/error
        # await delete_file(image_path)

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
    """
    Extracts text from an image using OCR.

    Args:
        image_path (str): The path to the image file.

    Returns:
        str: The extracted text.
    """
    processed_image_path = ""
    try:
        processed_image_path = await preprocess_image(image_path)

        # Use Tesseract to extract text
        text = pytesseract.image_to_string(Image.open(processed_image_path))

        # Delete file post success/error
        # await delete_file(processed_image_path)

        return text
    except Exception as exception:
        # Delete file post success/error
        await delete_file(processed_image_path)
        print(f"Error in extract_text: {repr(exception)}")


async def preprocess_image(image_path):
    """
    Preprocesses an image by converting it to grayscale and applying thresholding.

    Args:
        image_path (str): The path to the image file.

    Returns:
        str: The path to the processed image file.
    """
    try:
        # Read image using OpenCV
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)

        # Convert to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply thresholding
        # gray_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        # 1
        # gray_image = cv2.threshold(gray_image, 120, 255, cv2.THRESH_BINARY)[1]

        # 2
        # gray_image = cv2.threshold(gray_image, 120, 255, cv2.THRESH_BINARY_INV)[1]

        # 4
        gray_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_TOZERO)[1]

        # Save the processed image temporarily
        image_path = image_path.replace("uploads/", "")
        processed_image_path = (
            f"uploads/{os.path.splitext(image_path)[0]}_processed.png"
        )

        cv2.imwrite(processed_image_path, gray_image)

        return processed_image_path
    except Exception as exception:
        print(f"Error in preprocess_image: {repr(exception)}")


async def delete_file(file_path):
    """
    Deletes a file from the specified file path.

    Args:
        file_path (str): The path of the file to be deleted.

    Returns:
        bool: True if the file is successfully deleted, False otherwise.
    """
    try:
        # Delete processed image once text is extracted
        # Check if the file exists
        if os.path.exists(file_path):
            # Delete the file
            os.remove(file_path)

        return True
    except Exception as exception:
        print(f"Error in delete_file: {repr(exception)}")
        return False