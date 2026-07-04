import base64
import json
import os
import httpx
from flask import Flask, request, jsonify
import anthropic

app = Flask(__name__)

CLAUDE_API_KEY = os.environ.get("CLAUDE_API_KEY", "")

TOOL = {
    "name": "extract_documents",
    "description": (
        "מחלץ מידע מובנה מתוך מסמכי משכנתה (תלושי שכר, דפי עו\"ש). "
        "מחזיר רשימת מסמכים עם כל השדות הנדרשים."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "documents": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "document_type": {
                            "type": "string",
                            "description": "סוג המסמך, לדוגמה: תלוש שכר, דפי עו\"ש, שומת מס"
                        },
                        "pages": {
                            "type": "string",
                            "description": "מספר עמוד או טווח עמודים, לדוגמה: 1, 1-3"
                        },
                        "owner_name": {
                            "type": "string",
                            "description": "שם מלא של בעל המסמך"
                        },
                        "borrower": {
                            "type": "string",
                            "description": "זיהוי הלווה, לדוגמה: לווה 1, לווה 2, ערב"
                        },
                        "net_income": {
                            "type": "number",
                            "description": "הכנסה חודשית נטו בשקלים, אם רלוונטי למסמך"
                        },
                        "red_flags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "רשימת חריגות פיננסיות. אם תקין — [\"תקין\"]"
                        },
                        "red_flags_details": {
                            "type": "string",
                            "description": "תיאור מפורט בעברית של הממצאים, או \"תקין\""
                        }
                    },
                    "required": ["document_type", "pages", "owner_name", "borrower"]
                }
            }
        },
        "required": ["documents"]
    }
}


def analyze_pdf_bytes(pdf_bytes: bytes) -> dict:
    pdf_b64 = base64.standard_b64encode(pdf_bytes).decode("utf-8")
    client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)

    with client.messages.stream(
        model="claude-opus-4-8",
        max_tokens=4000,
        tools=[TOOL],
        tool_choice={"type": "tool", "name": "extract_documents"},
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": "application/pdf",
                        "data": pdf_b64
                    }
                },
                {
                    "type": "text",
                    "text": (
                        "נתח את מסמכי המשכנתה המצורפים עבור פרדס פיננסים. "
                        "עבור כל מסמך חלץ: סוג מסמך, עמודים, שם בעלים, זיהוי לווה, "
                        "הכנסה חודשית נטו (אם קיים), דגלים אדומים ופירוט. "
                        "אם אין חריגות — red_flags: [\"תקין\"], red_flags_details: \"תקין\"."
                    )
                }
            ]
        }]
    ) as stream:
        result = stream.get_final_message()

    for block in result.content:
        if block.type == "tool_use" and block.name == "extract_documents":
            return block.input

    return {"documents": []}


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Accepts either:
      - JSON body: {"pdf_url": "https://..."}
      - Multipart: file field named "file"
    Returns: {"documents": [...]}
    """
    try:
        # Option A: URL provided (Make.com passes a download URL)
        if request.is_json and request.json.get("pdf_url"):
            pdf_url = request.json["pdf_url"]
            resp = httpx.get(pdf_url, timeout=30, follow_redirects=True)
            resp.raise_for_status()
            pdf_bytes = resp.content

        # Option B: Raw file upload
        elif "file" in request.files:
            pdf_bytes = request.files["file"].read()

        # Option C: Raw binary body
        elif request.data:
            pdf_bytes = request.data

        else:
            return jsonify({"error": "Provide pdf_url in JSON body, a 'file' field, or raw binary body"}), 400

        result = analyze_pdf_bytes(pdf_bytes)
        return jsonify(result)

    except httpx.HTTPError as e:
        return jsonify({"error": f"Failed to download PDF: {str(e)}"}), 502
    except anthropic.APIError as e:
        return jsonify({"error": f"Claude API error: {str(e)}"}), 502
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
