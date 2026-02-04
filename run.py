#!/usr/bin/env python
"""Entry point to run the Universal Barcode Scanner API."""

import uvicorn
import sys

if __name__ == "__main__":
    print("=" * 60)
    print("Universal Barcode Scanner API")
    print("=" * 60)
    print("\nStarting API server...")
    print("Access the API at: http://localhost:8000")
    print("Interactive docs: http://localhost:8000/docs")
    print("ReDoc: http://localhost:8000/redoc")
    print("\nPress CTRL+C to stop the server\n")
    print("=" * 60)
    
    try:
        uvicorn.run(
            "app.main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n\nServer stopped.")
        sys.exit(0)
