import os
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional

from app.bot import process_bot_response, get_or_create_session, reset_session, QUICK_REPLIES
from app.vision import preprocess_image, run_defect_inference

app = FastAPI(title="E-commerce support Bot & Defect Detection API")

# Define request/response structures
class ChatRequest(BaseModel):
    session_id: str
    message: str

class ResetRequest(BaseModel):
    session_id: str

# Create path references
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(BASE_DIR, "static")

# Mount static folder
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serves the main application page."""
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return HTMLResponse("<h2>Frontend Index HTML file not found yet.</h2>")

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    """Processes conversational inputs."""
    try:
        bot_msg, quick_replies, prompt_upload = process_bot_response(
            request.session_id, request.message
        )
        session = get_or_create_session(request.session_id)
        
        return {
            "session_id": request.session_id,
            "response": bot_msg,
            "state": session["state"],
            "quick_replies": quick_replies,
            "prompt_upload": prompt_upload
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload")
async def upload_endpoint(
    session_id: str = Form(...),
    file: UploadFile = File(...)
):
    """Handles defect photo uploads and routes through the CV pipeline."""
    # Ensure it's an image
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")
        
    try:
        session = get_or_create_session(session_id)
        
        # Read file bytes safely with a context manager or direct stream read
        content = await file.read()
        
        # Save to uploads folder for visual inspection and debugging
        uploads_dir = os.path.join(BASE_DIR, "uploads")
        os.makedirs(uploads_dir, exist_ok=True)
        saved_file_path = os.path.join(uploads_dir, file.filename)
        with open(saved_file_path, "wb") as f:
            f.write(content)
        
        # 1. Preprocess using Pillow and NumPy
        img_normalized, stats, cv_logs = preprocess_image(content)
        
        # 2. Run mock inference pipeline
        reported_issue = session.get("reported_issue", "")
        inference_result = run_defect_inference(
            img_normalized, stats, file.filename, reported_issue
        )
        
        # Remove non-serializable NumPy arrays from stats before returning JSON
        if "img_segmentation" in stats:
            stats.pop("img_segmentation")
            
        # 3. Transition dialogue state based on CV verdict
        verified = inference_result["verified"]
        if verified:
            session["state"] = "DEFECT_VERIFIED"
            bot_text = (
                f"🔍 **Visual Verification Result:** Defect detected with "
                f"**{inference_result['confidence']*100:.1f}% confidence**.\n\n"
                f"**Details:** {inference_result['details']}\n\n"
                f"I have initialized the return processing protocol. Would you like me to generate your prepaid shipping label now?"
            )
        else:
            session["state"] = "DEFECT_REJECTED"
            bot_text = (
                f"🔍 **Visual Verification Result:** No defect detected "
                f"({inference_result['confidence']*100:.1f}% confidence, threshold: {inference_result['threshold']*100:.1f}%).\n\n"
                f"**Details:** {inference_result['details']}\n\n"
                f"If you'd like to try again, please upload a clearer, well-lit, close-up photo of the damage. "
                f"Alternatively, you can choose to transfer this chat to a human agent for manual review."
            )
            
        session["verdict_data"] = inference_result
        session["history"].append({"sender": "bot", "text": bot_text})
        
        return {
            "session_id": session_id,
            "verdict": inference_result["verdict"],
            "confidence": inference_result["confidence"],
            "details": inference_result["details"],
            "verified": verified,
            "stats": stats,
            "logs": cv_logs,
            "bot_response": bot_text,
            "next_state": session["state"],
            "quick_replies": QUICK_REPLIES[session["state"]]
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Image processing failed: {str(e)}")

@app.post("/api/reset")
async def reset_endpoint(request: ResetRequest):
    """Resets conversation state for a clean session."""
    session = reset_session(request.session_id)
    return {
        "session_id": request.session_id,
        "state": session["state"],
        "message": "Session reset successfully"
    }

@app.get("/api/session/{session_id}")
async def get_session_endpoint(session_id: str):
    """Retrieves current session data."""
    session = get_or_create_session(session_id)
    return {
        "session_id": session_id,
        "state": session["state"],
        "order_id": session["order_id"],
        "reported_issue": session["reported_issue"]
    }
