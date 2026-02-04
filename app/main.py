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

OCR_FIELD_MAP = {
    "FIRST_NAME": "first_name",
    "MIDDLE_NAME": "middle_name",
    "LAST_NAME": "last_name",
    "SUFFIX": "suffix",
    "DATE_OF_BIRTH": "dob",
    "DATE_OF_ISSUE": "issue_date",
    "EXPIRATION_DATE": "expiry_date",
    "DOCUMENT_NUMBER": "license_number",
    "ID_TYPE": "id_type",
    "ADDRESS": "street",
    "CITY_IN_ADDRESS": "city",
    "STATE_IN_ADDRESS": "state",
    "STATE_NAME": "state_name",
    "ZIP_CODE_IN_ADDRESS": "postal_code",
    "ENDORSEMENTS": "endorsements",
    "RESTRICTIONS": "restrictions",
    "CLASS": "class",
    "VETERAN": "veteran",
    "COUNTY": "county",
    "PLACE_OF_BIRTH": "place_of_birth",
    "MRZ_CODE": "mrz_code",
}


def _to_snake(text: str) -> str:
    return text.strip().lower().replace(" ", "_")


def _to_camel(text: str) -> str:
    parts = text.strip().split("_")
    if not parts:
        return text
    return parts[0] + "".join(p[:1].upper() + p[1:] for p in parts[1:])


def _parse_barcode_confidence(barcode: dict) -> float:
    confidence = (
        barcode.get("meta", {}) or {}
    ).get("confidence")
    if isinstance(confidence, str) and confidence.endswith("%"):
        try:
            return float(confidence.rstrip("%"))
        except ValueError:
            return 100.0
    if isinstance(confidence, (int, float)):
        return float(confidence)
    return 100.0


def _build_barcode_field_map(pdf417_barcode: dict) -> dict:
    if not pdf417_barcode:
        return {}
    confidence = _parse_barcode_confidence(pdf417_barcode)
    fields = {}
    for section in ["person", "document", "address"]:
        data = pdf417_barcode.get(section) or {}
        for key, value in data.items():
            fields[key] = {
                "value": value,
                "confidence": confidence,
            }
    return fields


def _build_ocr_field_map(ocr_fields: list) -> dict:
    fields = {}
    for item in ocr_fields or []:
        field_type = ((item.get("Type") or {}).get("Text") or "").strip()
        if not field_type:
            continue
        key = OCR_FIELD_MAP.get(field_type, _to_snake(field_type))
        value = (item.get("ValueDetection") or {}).get("Text")
        confidence = (item.get("ValueDetection") or {}).get("Confidence")
        fields[key] = {
            "value": value,
            "confidence": confidence,
        }
    return fields


def _flatten_ocr_docs(ocr_docs: list) -> list:
    fields = []
    for doc in ocr_docs or []:
        fields.extend(doc.get("IdentityDocumentFields", []) or [])
    return fields


def _extract_id_type_value(fields: list) -> str | None:
    for item in fields or []:
        field_type = ((item.get("Type") or {}).get("Text") or "").strip()
        if field_type == "ID_TYPE":
            return (item.get("ValueDetection") or {}).get("Text")
    return None


def _group_ocr_docs(ocr_docs: list) -> dict:
    groups = {"front": [], "back": [], "unknown": []}
    for doc in ocr_docs or []:
        fields = doc.get("IdentityDocumentFields", []) or []
        id_type = (_extract_id_type_value(fields) or "").upper()
        if "FRONT" in id_type:
            groups["front"].extend(fields)
        elif "BACK" in id_type:
            groups["back"].extend(fields)
        else:
            groups["unknown"].extend(fields)
    return groups


def _ocr_map_with_address(fields: list) -> dict:
    field_map = _build_ocr_field_map(fields)
    out = {}
    for key, payload in field_map.items():
        out[_to_camel(key)] = {
            "value": payload.get("value"),
            "confidence": payload.get("confidence"),
        }
    address = {
        "street": field_map.get("street", {}).get("value"),
        "city": field_map.get("city", {}).get("value"),
        "state": field_map.get("state", {}).get("value"),
        "postalCode": field_map.get("postal_code", {}).get("value"),
    }
    if any(address.values()):
        out["address"] = address
    return out


def _score(confidence) -> float:
    if isinstance(confidence, (int, float)):
        return float(confidence)
    return 0.0


def _final_fields(barcode_fields: dict, ocr_fields: dict) -> dict:
    final = {}
    all_keys = set(barcode_fields.keys()) | set(ocr_fields.keys())
    for key in sorted(all_keys):
        b = barcode_fields.get(key) or {}
        o = ocr_fields.get(key) or {}
        b_val = b.get("value")
        o_val = o.get("value")
        b_score = _score(b.get("confidence")) if b_val not in (None, "") else 0.0
        o_score = _score(o.get("confidence")) if o_val not in (None, "") else 0.0
        if b_score == 0.0 and o_score == 0.0:
            continue
        if b_score >= o_score:
            final[key] = b_val
        else:
            final[key] = o_val
    return final


def _barcode_data(pdf417_barcodes: list) -> dict:
    if not pdf417_barcodes:
        return {"detected": False}
    barcode = pdf417_barcodes[0]
    confidence = (barcode.get("meta") or {}).get("confidence")
    return {
        "detected": True,
        "type": barcode.get("type"),
        "confidence": confidence,
        "raw": barcode.get("raw"),
        "person": {
            "firstName": (barcode.get("person") or {}).get("first_name"),
            "middleName": (barcode.get("person") or {}).get("middle_name"),
            "lastName": (barcode.get("person") or {}).get("last_name"),
            "suffix": (barcode.get("person") or {}).get("suffix"),
            "sex": (barcode.get("person") or {}).get("sex"),
            "eyeColor": (barcode.get("person") or {}).get("eye_color"),
            "hairColor": (barcode.get("person") or {}).get("hair_color"),
            "heightIn": (barcode.get("person") or {}).get("height_in"),
            "weightLb": (barcode.get("person") or {}).get("weight_lb"),
        },
        "document": {
            "licenseNumber": (barcode.get("document") or {}).get("license_number"),
            "issueDate": (barcode.get("document") or {}).get("issue_date"),
            "expiryDate": (barcode.get("document") or {}).get("expiry_date"),
            "dob": (barcode.get("document") or {}).get("dob"),
            "issuerCountry": (barcode.get("document") or {}).get("issuer_country"),
            "auditNumber": (barcode.get("document") or {}).get("audit_number"),
            "documentDiscriminator": (barcode.get("document") or {}).get("document_discriminator"),
            "cardRevisionDate": (barcode.get("document") or {}).get("card_revision_date"),
        },
        "address": {
            "street": (barcode.get("address") or {}).get("street"),
            "city": (barcode.get("address") or {}).get("city"),
            "state": (barcode.get("address") or {}).get("state"),
            "postalCode": (barcode.get("address") or {}).get("postal_code"),
        },
    }


def _normalize_id_type(value: str | None) -> str:
    if not value:
        return "UNKNOWN"
    upper = value.upper()
    if "DRIVER" in upper and "LICENSE" in upper:
        return "DRIVER_LICENSE"
    return "UNKNOWN"


def _parse_date(value: str | None):
    if not value:
        return None
    try:
        from datetime import datetime
        return datetime.strptime(value, "%Y-%m-%d").date()
    except Exception:
        return None


def _structured_data(final_fields: dict, source_priority: str, id_type_value: str | None) -> dict:
    expiry_date = final_fields.get("expiry_date")
    parsed_expiry = _parse_date(expiry_date)
    today = _parse_date("2026-02-04")
    is_expired = False
    if parsed_expiry and today:
        is_expired = parsed_expiry < today

    confidence = "HIGH" if source_priority == "BARCODE" else "MEDIUM"
    if not final_fields:
        confidence = "LOW"

    def _pick(mapping: dict) -> dict:
        out = {}
        for out_key, in_key in mapping.items():
            value = final_fields.get(in_key)
            if value not in (None, ""):
                out[out_key] = value
        return out

    person_map = {
        "firstName": "first_name",
        "middleName": "middle_name",
        "lastName": "last_name",
        "dob": "dob",
        "sex": "sex",
    }
    document_map = {
        "licenseNumber": "license_number",
        "issueDate": "issue_date",
        "expiryDate": "expiry_date",
        "class": "class",
        "endorsements": "endorsements",
        "restrictions": "restrictions",
        "issuerCountry": "issuer_country",
        "auditNumber": "audit_number",
        "documentDiscriminator": "document_discriminator",
        "cardRevisionDate": "card_revision_date",
        "hazmatExpiry": "hazmat_expiry",
        "idType": "id_type",
    }
    address_map = {
        "street": "street",
        "city": "city",
        "state": "state",
        "postalCode": "postal_code",
    }
    physical_map = {
        "eyeColor": "eye_color",
        "hairColor": "hair_color",
        "heightIn": "height_in",
        "weightLb": "weight_lb",
    }

    person = _pick(person_map)
    document = _pick(document_map)
    address = _pick(address_map)
    physical = _pick(physical_map)

    structured = {
        "idType": _normalize_id_type(id_type_value),
        "sourcePriority": source_priority,
    }
    if person:
        structured["person"] = person
    if document:
        structured["document"] = document
    if address:
        structured["address"] = address
    if physical:
        structured["physicalAttributes"] = physical

    known_keys = set(person_map.values()) | set(document_map.values()) | set(address_map.values()) | set(physical_map.values())
    extra_fields = {}
    for key, value in final_fields.items():
        if key in known_keys or value in (None, ""):
            continue
        extra_fields[_to_camel(key)] = value
    if extra_fields:
        structured["extraFields"] = extra_fields
    structured["meta"] = {
        "isExpired": is_expired,
        "expiryDate": expiry_date,
        "confidence": confidence,
    }
    return structured

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
        textract_id_docs = OcrReader.extract_identity_document_fields(contents)
        all_barcodes = barcode_result.get("barcodes", [])
        any_barcode_detected = bool(barcode_result.get("success")) and bool(all_barcodes)
        pdf417_barcodes = [
            b for b in all_barcodes
            if str(b.get("type", "")).upper() == "PDF417"
        ]
        barcode_fields = _build_barcode_field_map(
            pdf417_barcodes[0] if pdf417_barcodes else None
        )
        ocr_flat_fields = _flatten_ocr_docs(textract_id_docs)
        ocr_fields = _build_ocr_field_map(ocr_flat_fields)
        final_fields = _final_fields(barcode_fields, ocr_fields)
        ocr_groups = _group_ocr_docs(textract_id_docs)
        id_type_value = _extract_id_type_value(ocr_groups.get("front") or []) or _extract_id_type_value(ocr_groups.get("back") or [])
        source_priority = "BARCODE" if pdf417_barcodes else "OCR"
        structured = _structured_data(final_fields, source_priority, id_type_value)
        barcode_data = _barcode_data(pdf417_barcodes)
        ocr_data = {
            "detected": bool(ocr_flat_fields),
            "front": _ocr_map_with_address(ocr_groups.get("front")),
            "back": _ocr_map_with_address(ocr_groups.get("back")),
        }

        if not any_barcode_detected:
            return JSONResponse(
                status_code=200,
                content={
                    "success": True,
                    "totalFiles": 1,
                    "barcodeData": {"detected": False},
                    "ocrData": ocr_data,
                    "structuredData": structured,
                }
            )

        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "totalFiles": 1,
                "barcodeData": barcode_data,
                "ocrData": ocr_data,
                "structuredData": structured,
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
                textract_id_docs = OcrReader.extract_identity_document_fields(contents)
                all_barcodes = barcode_result.get("barcodes", [])
                any_barcode_detected = bool(barcode_result.get("success")) and bool(all_barcodes)
                pdf417_barcodes = [
                    b for b in all_barcodes
                    if str(b.get("type", "")).upper() == "PDF417"
                ]
                barcode_fields = _build_barcode_field_map(
                    pdf417_barcodes[0] if pdf417_barcodes else None
                )
                ocr_flat_fields = _flatten_ocr_docs(textract_id_docs)
                ocr_fields = _build_ocr_field_map(ocr_flat_fields)
                final_fields = _final_fields(barcode_fields, ocr_fields)
                ocr_groups = _group_ocr_docs(textract_id_docs)
                id_type_value = _extract_id_type_value(ocr_groups.get("front") or []) or _extract_id_type_value(ocr_groups.get("back") or [])
                source_priority = "BARCODE" if pdf417_barcodes else "OCR"
                structured = _structured_data(final_fields, source_priority, id_type_value)
                barcode_data = _barcode_data(pdf417_barcodes)
                ocr_data = {
                    "detected": bool(ocr_flat_fields),
                    "front": _ocr_map_with_address(ocr_groups.get("front")),
                    "back": _ocr_map_with_address(ocr_groups.get("back")),
                }

                if not any_barcode_detected:
                    results.append({
                        "filename": file.filename,
                        "barcodeData": {"detected": False},
                        "ocrData": ocr_data,
                        "structuredData": structured,
                    })
                else:
                    results.append({
                        "filename": file.filename,
                        "barcodeData": barcode_data,
                        "ocrData": ocr_data,
                        "structuredData": structured,
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
                    "barcodeData": {"detected": False},
                    "ocrData": {
                        "detected": False,
                        "front": {},
                        "back": {},
                    },
                    "structuredData": {}
                })
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "totalFiles": len(files),
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
