"""Main FastAPI application for barcode scanning."""

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
from PIL import Image
from app.config import API_TITLE, API_VERSION, API_DESCRIPTION, ALLOWED_EXTENSIONS, MAX_FILE_SIZE
from app.utils.barcode_reader import BarcodeReader
from app.utils.ocr_reader import OcrReader

# Load local environment variables from .env if present
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=API_DESCRIPTION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Health"])
async def root():
    """Root endpoint for health check."""
    return {
        "status": "active",
        "service": API_TITLE,
        "version": API_VERSION,
        "message": "API is running and ready to extract barcodes"
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": API_TITLE,
        "version": API_VERSION
    }


@app.post("/extract-barcode", tags=["Barcode Extraction"])
async def extract_barcode(file: UploadFile = File(...)):
    """
    Extract barcode information from an uploaded image.
    
    Args:
        file: Image file with barcode(s)
        
    Returns:
        JSON response with extracted barcode information
    """
    try:
        # Validate file extension
        if file.filename:
            file_ext = file.filename.split(".")[-1].lower()
            if file_ext not in ALLOWED_EXTENSIONS:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid file type. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
                )
        
        # Read file content
        contents = await file.read()
        
        # Validate file size
        if len(contents) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File size exceeds maximum limit of {MAX_FILE_SIZE / 1024 / 1024}MB"
            )
        
        # Validate image
        is_valid, validation_message = BarcodeReader.validate_image(contents)
        if not is_valid:
            raise HTTPException(
                status_code=400,
                detail=validation_message
            )
        
        # Extract barcodes and OCR
        barcode_result = BarcodeReader.extract_barcodes(contents)
        textract_id_fields = OcrReader.extract_identity_document_fields(contents)
        all_barcodes = barcode_result.get("barcodes", [])
        any_barcode_detected = bool(barcode_result.get("success")) and bool(all_barcodes)
        pdf417_barcodes = [
            b for b in all_barcodes
            if str(b.get("type", "")).upper() == "PDF417"
        ]
        pdf417_detected = bool(pdf417_barcodes)

        if not any_barcode_detected:
            return JSONResponse(
                status_code=200,
                content={
                    "ocr": {
                        "detected": bool(textract_id_fields),
                        "data": textract_id_fields,
                    }
                }
            )

        return JSONResponse(
            status_code=200,
            content={
                "pdf417": {
                    "detected": pdf417_detected,
                    "message": None if pdf417_detected else "PDF417 not detected",
                    "data": pdf417_barcodes,
                },
                "barcode": {
                    "detected": True,
                    "data": all_barcodes,
                },
                "ocr": {
                    "detected": bool(textract_id_fields),
                    "data": textract_id_fields,
                },
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": f"Internal server error: {str(e)}",
                "barcodes": []
            }
        )


@app.post("/extract-barcode-batch", tags=["Barcode Extraction"])
async def extract_barcode_batch(files: list[UploadFile] = File(...)):
    """
    Extract barcode information from multiple uploaded images.
    
    Args:
        files: List of image files with barcode(s)
        
    Returns:
        JSON response with extracted barcode information from all images
    """
    try:
        results = []
        
        for file in files:
            try:
                # Validate file extension
                if file.filename:
                    file_ext = file.filename.split(".")[-1].lower()
                    if file_ext not in ALLOWED_EXTENSIONS:
                        results.append({
                            "filename": file.filename,
                            "success": False,
                            "message": f"Invalid file type. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}",
                            "barcode": {
                                "success": False,
                                "message": f"Invalid file type. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}",
                                "count": 0,
                                "barcodes": []
                            },
                            "ocr": {
                                "success": False,
                                "message": "OCR not available for invalid files.",
                                "text": "",
                                "lines": []
                            }
                        })
                        continue
                
                # Read file content
                contents = await file.read()
                
                # Validate file size
                if len(contents) > MAX_FILE_SIZE:
                    results.append({
                        "filename": file.filename,
                        "success": False,
                        "message": f"File size exceeds maximum limit of {MAX_FILE_SIZE / 1024 / 1024}MB",
                        "barcode": {
                            "success": False,
                            "message": f"File size exceeds maximum limit of {MAX_FILE_SIZE / 1024 / 1024}MB",
                            "count": 0,
                            "barcodes": []
                        },
                        "ocr": {
                            "success": False,
                            "message": "OCR not available for oversized files.",
                            "text": "",
                            "lines": []
                        }
                    })
                    continue
                
                # Validate image
                is_valid, validation_message = BarcodeReader.validate_image(contents)
                if not is_valid:
                    results.append({
                        "filename": file.filename,
                        "success": False,
                        "message": validation_message,
                        "barcode": {
                            "success": False,
                            "message": validation_message,
                            "count": 0,
                            "barcodes": []
                        },
                        "ocr": {
                            "success": False,
                            "message": "OCR not available for invalid images.",
                            "text": "",
                            "lines": []
                        }
                    })
                    continue
                
                # Extract barcodes and OCR
                barcode_result = BarcodeReader.extract_barcodes(contents)
                textract_id_fields = OcrReader.extract_identity_document_fields(contents)
                all_barcodes = barcode_result.get("barcodes", [])
                any_barcode_detected = bool(barcode_result.get("success")) and bool(all_barcodes)
                pdf417_barcodes = [
                    b for b in all_barcodes
                    if str(b.get("type", "")).upper() == "PDF417"
                ]
                pdf417_detected = bool(pdf417_barcodes)

                if not any_barcode_detected:
                    results.append({
                        "filename": file.filename,
                        "ocr": {
                            "detected": bool(textract_id_fields),
                            "data": textract_id_fields,
                        },
                    })
                else:
                    results.append({
                        "filename": file.filename,
                        "pdf417": {
                            "detected": pdf417_detected,
                            "message": None if pdf417_detected else "PDF417 not detected",
                            "data": pdf417_barcodes,
                        },
                        "barcode": {
                            "detected": True,
                            "data": all_barcodes,
                        },
                        "ocr": {
                            "detected": bool(textract_id_fields),
                            "data": textract_id_fields,
                        },
                    })
                
            except Exception as e:
                results.append({
                    "filename": file.filename,
                    "success": False,
                    "message": f"Error processing file: {str(e)}",
                    "pdf417": {
                        "detected": False,
                        "message": f"Error processing file: {str(e)}",
                        "data": []
                    },
                    "ocr": {
                        "detected": False,
                        "data": []
                    }
                })
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "total_files": len(files),
                "results": results
            }
        )
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": f"Internal server error: {str(e)}",
                "results": []
            }
        )


@app.post("/debug/barcode", tags=["Debug"])
async def debug_barcode(file: UploadFile = File(...)):
    """
    Debug endpoint to inspect decoder availability and image details.
    """
    try:
        contents = await file.read()
        is_valid, validation_message = BarcodeReader.validate_image(contents)
        if not is_valid:
            raise HTTPException(status_code=400, detail=validation_message)

        image = Image.open(BytesIO(contents)).convert("RGB")
        width, height = image.size

        # Report decoder availability from module globals
        from app.utils import barcode_reader as br
        decoder_status = {
            "zxingcpp": getattr(br, "ZXING_AVAILABLE", False),
        }

        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "image": {
                    "width": width,
                    "height": height,
                    "mode": image.mode,
                    "format": image.format,
                },
                "decoders": decoder_status,
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": f"Internal server error: {str(e)}",
            }
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
