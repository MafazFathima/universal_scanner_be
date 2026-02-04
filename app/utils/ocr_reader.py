import os
import logging
import boto3
from botocore.exceptions import BotoCoreError, ClientError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OcrReader:
    @staticmethod
    @staticmethod
    def _get_textract_client():
        region = os.getenv("AWS_REGION", "us-east-1")
        endpoint = os.getenv("AWS_TEXTRACT_ENDPOINT", "")
        access_key = os.getenv("AWS_ACCESS_KEY_ID", "")
        secret_key = os.getenv("AWS_SECRET_ACCESS_KEY", "")

        if not access_key or not secret_key:
            return None, "AWS credentials are not configured"

        kwargs = {
            "service_name": "textract",
            "region_name": region,
            "aws_access_key_id": access_key,
            "aws_secret_access_key": secret_key,
        }
        if endpoint:
            kwargs["endpoint_url"] = endpoint

        return boto3.client(**kwargs), None

    @staticmethod
    def extract_text(image_bytes: bytes) -> dict:
        client, error = OcrReader._get_textract_client()
        if error:
            return {
                "success": False,
                "message": "OCR not available. Please contact support.",
                "raw": None,
            }
        try:
            response = client.detect_document_text(
                Document={"Bytes": image_bytes}
            )
            return {
                "success": True,
                "message": "OCR completed",
                "raw": response,
            }
        except (BotoCoreError, ClientError) as e:
            logger.exception("OCR failed")
            return {
                "success": False,
                "message": "OCR failed. Please contact support.",
                "raw": None,
            }

    @staticmethod
    def extract_identity_document_fields(image_bytes: bytes) -> list:
        client, error = OcrReader._get_textract_client()
        if error:
            return []
        try:
            response = client.analyze_id(
                DocumentPages=[{"Bytes": image_bytes}]
            )
            identity_documents = response.get("IdentityDocuments", []) or []
            # Return only the IdentityDocumentFields array per document (no blocks)
            cleaned = []
            for doc in identity_documents:
                doc_fields = doc.get("IdentityDocumentFields", []) or []
                cleaned.append({
                    "DocumentIndex": doc.get("DocumentIndex"),
                    "IdentityDocumentFields": doc_fields,
                })
            return cleaned
        except (BotoCoreError, ClientError) as e:
            logger.exception("Textract AnalyzeID failed")
            return []
