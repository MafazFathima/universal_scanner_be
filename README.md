# Universal Barcode Scanner API

A FastAPI-based REST API for extracting barcode information from images. The API accepts image files as payloads, reads barcodes, and returns extracted barcode data in JSON format.

## Features

- **Single Image Barcode Extraction**: Upload a single image and extract barcode information
- **Batch Processing**: Process multiple images in a single request
- **Multiple Barcode Types Supported**: Works with various barcode formats (UPC, EAN, QR codes, etc.)
- **Image Validation**: Validates image format and size before processing
- **CORS Enabled**: Ready for cross-origin requests
- **Comprehensive Error Handling**: Detailed error messages and validation feedback

## Project Structure

```
universal_scanner/
├── app/
│   ├── __init__.py
│   ├── main.py                 # Main FastAPI application
│   ├── config.py               # Configuration settings
│   └── utils/
│       ├── __init__.py
│       └── barcode_reader.py   # Barcode reading utilities
├── tests/                       # Test files directory
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Installation

1. **Clone or navigate to the project directory**:
   ```bash
   cd universal_scanner
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/Scripts/activate  # On Windows
   source venv/bin/activate      # On macOS/Linux
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Running the API

Start the API server using:

```bash
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Or directly:

```bash
python app/main.py
```

The API will be available at `http://localhost:8000`

## API Documentation

Once the server is running, access the interactive API documentation:

- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

## API Endpoints

### 1. Health Check

**GET** `/` or `/health`

Check if the API is running.

**Response**:
```json
{
  "status": "active",
  "service": "Universal Barcode Scanner API",
  "version": "1.0.0",
  "message": "API is running and ready to extract barcodes"
}
```

### 2. Extract Barcode from Single Image

**POST** `/extract-barcode`

Extract barcode information from a single image file.

**Request**:
- **Method**: POST
- **Content-Type**: multipart/form-data
- **Parameter**: `file` (required) - Image file (jpg, jpeg, png, gif, bmp)
- **Max File Size**: 10MB

**Example using cURL**:
```bash
curl -X POST "http://localhost:8000/extract-barcode" \
  -H "accept: application/json" \
  -F "file=@/path/to/barcode_image.jpg"
```

**Example using Python**:
```python
import requests

with open('barcode_image.jpg', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:8000/extract-barcode', files=files)
    print(response.json())
```

**Response - Success**:
```json
{
  "success": true,
  "message": "Successfully extracted 2 barcode(s)",
  "count": 2,
  "barcodes": [
    {
      "type": "EAN13",
      "data": "5901234123457",
      "quality": "readable",
      "rect": {
        "x": 100,
        "y": 150,
        "width": 200,
        "height": 80
      }
    },
    {
      "type": "CODE128",
      "data": "ABC123456789",
      "quality": "readable",
      "rect": {
        "x": 350,
        "y": 200,
        "width": 180,
        "height": 75
      }
    }
  ]
}
```

**Response - No Barcodes Found**:
```json
{
  "success": false,
  "message": "No barcodes found in the image",
  "barcodes": []
}
```

### 3. Extract Barcodes from Multiple Images

**POST** `/extract-barcode-batch`

Process multiple image files in a single request.

**Request**:
- **Method**: POST
- **Content-Type**: multipart/form-data
- **Parameter**: `files` (required) - Multiple image files
- **Max File Size**: 10MB per file

**Example using cURL**:
```bash
curl -X POST "http://localhost:8000/extract-barcode-batch" \
  -H "accept: application/json" \
  -F "files=@/path/to/image1.jpg" \
  -F "files=@/path/to/image2.png"
```

**Example using Python**:
```python
import requests

files = [
    ('files', open('image1.jpg', 'rb')),
    ('files', open('image2.png', 'rb'))
]
response = requests.post('http://localhost:8000/extract-barcode-batch', files=files)
print(response.json())
```

**Response**:
```json
{
  "success": true,
  "total_files": 2,
  "results": [
    {
      "filename": "image1.jpg",
      "success": true,
      "message": "Successfully extracted 1 barcode(s)",
      "count": 1,
      "barcodes": [
        {
          "type": "QR_CODE",
          "data": "https://example.com",
          "quality": "readable",
          "rect": {
            "x": 50,
            "y": 50,
            "width": 300,
            "height": 300
          }
        }
      ]
    },
    {
      "filename": "image2.png",
      "success": false,
      "message": "No barcodes found in the image",
      "barcodes": []
    }
  ]
}
```

## Supported Barcode Formats

The API supports the following barcode formats:

- UPC (A & E)
- EAN (8 & 13)
- CODE128
- CODE39
- ITF
- QR Code
- PDF417
- Datamatrix
- And more...

## Configuration

Edit [app/config.py](app/config.py) to customize:

- API title, version, and description
- Maximum file size limit
- Allowed file extensions
- Temporary upload directory

## Error Handling

The API provides detailed error messages for various scenarios:

| Status Code | Error | Description |
|---|---|---|
| 200 | Success | Barcode extraction completed |
| 400 | Invalid File | File format not supported or image validation failed |
| 413 | File Too Large | File exceeds 10MB limit |
| 500 | Server Error | Internal server error during processing |

## Dependencies

- **FastAPI**: Modern web framework for building APIs
- **Uvicorn**: ASGI server
- **Pillow**: Image processing
- **OpenCV**: Computer vision library
- **ZXing-C++**: PDF417 barcode decoding
- **NumPy**: Numerical computing

## License

This project is provided as-is for educational and commercial use.

## Support

For issues or questions, please refer to the project repository or documentation.
