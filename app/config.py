"""Configuration settings for the barcode scanner API."""

# API Configuration
API_TITLE = "Universal Barcode Scanner API"
API_VERSION = "1.0.0"
API_DESCRIPTION = "API to extract barcode information from images"

# Upload Configuration
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "gif", "bmp"}
UPLOAD_TEMP_DIR = "temp"
