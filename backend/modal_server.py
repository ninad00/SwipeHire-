import os
import json
import tempfile
import numpy as np
import pytesseract
from pdf2image import convert_from_path

import modal
from google import genai
from google.genai import types
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import Depends, FastAPI, Header, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import firebase_admin
from firebase_admin import credentials, firestore, auth
from google.api_core.exceptions import ResourceExhausted

app = modal.App("tfj-backend")

volume = modal.Volume.from_name("tfj-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("tesseract-ocr", "poppler-utils", "libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "fastapi",
        "uvicorn[standard]",
        "python-multipart",
        "firebase-admin",
        "google-genai",
        "numpy",
        "scikit-learn",
        "pytesseract",
        "pdf2image",
        "python-dotenv",
    )
)

VOLUME_PATH = "/data"
FIREBASE_CRED_PATH = f"{VOLUME_PATH}/firebase-credentials.json"
SYSTEM_PROMPT_PATH = f"{VOLUME_PATH}/system_prompt.txt"

web_app = FastAPI(title="Resume Parser & Job Recommendation API")

web_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

firebase_initialized = False
db = None

def init_firebase():
    global firebase_initialized, db
    if not firebase_initialized:
        cred = credentials.Certificate(FIREBASE_CRED_PATH)
        firebase_admin.initialize_app(cred)
        db = firestore.client()
        firebase_initialized = True
    return db

class RankedJob(BaseModel):
    id: str
    title: str
    company: str
    tags: List[str]
    location: str
    date_posted: str
    apply_link: str
    description_snippet: str
    score: float

def get_current_user(authorization: str = Header(...)):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid auth header")
    token = authorization.split(" ")[1]
    try:
        decoded_token = auth.verify_id_token(token)
        return {
            "uid": decoded_token.get("uid"),
            "email": decoded_token.get("email"),
            "name": decoded_token.get("name"),
        }
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")

def extract_text_from_pdf(pdf_path: str) -> str:
    try:
        images = convert_from_path(pdf_path)
        text_content = ""
        for image in images:
            text_content += pytesseract.image_to_string(image)
        return text_content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF extraction error: {str(e)}")

def parse_resume_with_llm(text_content: str) -> dict:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not found")
    
    if not os.path.exists(SYSTEM_PROMPT_PATH):
        raise HTTPException(status_code=500, detail="system_prompt.txt not found")

    with open(SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as f:
        system_prompt = f.read()

    client = genai.Client(api_key=api_key)
    try:
        prompt = f"{system_prompt}\n\nResume Text:\n{text_content}"
        response = client.models.generate_content(
            model="gemini-2.0-flash", 
            contents=prompt,
        )
        response_text = response.text.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        return json.loads(response_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM parsing error: {str(e)}")

def create_embedding(text: str) -> np.ndarray:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not found")
    client = genai.Client(api_key=api_key)
    try:
        response = client.models.embed_content(
            model="gemini-embedding-2",
            contents=text,
            config=types.EmbedContentConfig(output_dimensionality=768)
        )
        return np.array(response.embeddings[0].values).reshape(1, -1)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding error: {str(e)}")

def load_job_database() -> List[dict]:
    try:
        database = init_firebase()
        jobs_ref = database.collection("resumes").stream()
        jobs = []
        for doc in jobs_ref:
            data = doc.to_dict()
            data["id"] = doc.id
            jobs.append(data)
        if not jobs:
            raise HTTPException(status_code=404, detail="No jobs found in Firestore")
        return jobs
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Firestore error: {str(e)}")

def rank_jobs_by_similarity(job_dict: dict, database: List[dict], top_k: int = 50) -> List[dict]:
    job_text = json.dumps(job_dict, sort_keys=True)
    query_embedding = create_embedding(job_text)
    results = []
    for item in database:
        embedding = item.get("embedding")
        if not embedding:
            continue
        item_embedding = np.array(embedding).reshape(1, -1)
        score = cosine_similarity(query_embedding, item_embedding)[0][0]
        job_result = {
            "id": item["id"],
            "title": item.get("title", ""),
            "company": item.get("company_name", ""),
            "tags": item.get("tags", []),
            "location": item.get("location", ""),
            "extensions": item.get("extensions", {}),
            "apply_link": item.get("share_link", ""),
            "description_snippet": item.get("description", "")[:300],
            "score": float(score),
        }
        results.append(job_result)
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]

def compute_and_store_ranking(uid):
    try:
        database = init_firebase()
        user_doc = database.collection("users").document(uid).get()
        if not user_doc.exists:
            return
        user_data = user_doc.to_dict()
        job_dict = user_data.get("job_dict")
        if not job_dict:
            return
        job_database = load_job_database()
        ranked_jobs = rank_jobs_by_similarity(job_dict, job_database, top_k=3000)
        seen_titles = set()
        unique_ranked_jobs = []
        for job in ranked_jobs:
            title = job.get("title", "").strip().lower()
            if title and title not in seen_titles:
                seen_titles.add(title)
                unique_ranked_jobs.append(job)
        ranked_jobs_data = [{"id": job["id"], "score": job["score"]} for job in unique_ranked_jobs]
        database.collection("users").document(uid).update({
            "ranked_jobs": ranked_jobs_data,
            "ranking_updated_at": firestore.SERVER_TIMESTAMP
        })
    except Exception as e:
        print(f"Error in compute_and_store_ranking: {str(e)}")

@web_app.get("/save-profile")
async def save_profile(user: dict = Depends(get_current_user)):
    try:
        database = init_firebase()
        user_doc = database.collection("users").document(user["uid"]).get()
        if not user_doc.exists:
            raise HTTPException(status_code=404, detail="User data not found")
        user_data = user_doc.to_dict()
        count = user_data.get("count", 0)
        ranked_jobs_data = user_data.get("ranked_jobs", [])
        current_batch = ranked_jobs_data[count : count + 5]
        jobs_to_send_details = []
        for item in current_batch:
            job_id = item["id"]
            score = item["score"]
            job_doc = database.collection("resumes").document(job_id).get()
            if job_doc.exists:
                job_data = job_doc.to_dict()
                job_data = {key: job_data.get(key, "") for key in ["apply_options", "company_name", "description", "detected_extensions", "extensions", "job_highlights", "location", "title"]}
                job_data["id"] = job_id
                job_data["score"] = score
                jobs_to_send_details.append(job_data)
        if jobs_to_send_details:
            database.collection("users").document(user["uid"]).update({
                "count": count + len(jobs_to_send_details)
            })
        return {
            "success": True,
            "ranked_jobs": jobs_to_send_details,
            "total_jobs": len(ranked_jobs_data)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@web_app.websocket("/ws/jobs")
async def jobs_ws(ws: WebSocket):
    await ws.accept()
    try:
        token = ws.query_params.get("token")
        if not token:
            await ws.close(code=1008)
            return
        decoded = auth.verify_id_token(token)
        uid = decoded["uid"]
        database = init_firebase()
        user_ref = database.collection("users").document(uid)
        user = user_ref.get().to_dict()
        ranked_jobs_data = user.get("ranked_jobs", [])
        count = user.get("count", 0)
        while True:
            data = json.loads(await ws.receive_text())
            if data["type"] == "NEXT_JOB":
                if count >= len(ranked_jobs_data):
                    await ws.send_json({"type": "END"})
                    continue
                item = ranked_jobs_data[count]
                job_id = item["id"]
                score = item["score"]
                count += 1
                doc = database.collection("resumes").document(job_id).get()
                if doc.exists:
                    job_data = doc.to_dict()
                    job_data = {key: job_data.get(key, "") for key in ["apply_options", "company_name", "description", "detected_extensions", "extensions", "job_highlights", "location", "title"]}
                    job_data["id"] = job_id
                    job_data["score"] = score
                    await ws.send_json({"type": "JOB", "job": job_data})
                user_ref.update({"count": count})
    except WebSocketDisconnect:
        pass

@web_app.post("/parse-resume")
async def parse_resume(file: UploadFile = File(...), user: dict = Depends(get_current_user)):
    if not file.filename.lower().endswith(('.pdf', '.docx')):
        raise HTTPException(status_code=400, detail="Only PDF and DOCX files are supported")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        tmp_path = tmp_file.name
    
    try:
        raw_text = extract_text_from_pdf(tmp_path)
        if not raw_text.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from PDF")
        parsed_data = parse_resume_with_llm(raw_text)
        info_dict = parsed_data.get("info_dict", {})
        job_dict = parsed_data.get("job_dict", {})
        new_keys_tracker = parsed_data.get("new_keys_tracker", {})
        database = init_firebase()
        database.collection("users").document(user["uid"]).set(
            {
                "info_dict": info_dict,
                "job_dict": job_dict,
                "dynamic_keys": new_keys_tracker,
                "count": 0,
                "ranked_jobs": [],
                "updated_at": firestore.SERVER_TIMESTAMP,
            },
            merge=False,
        )
        return {
            "success": True,
            "info_dict": info_dict,
            "job_dict": job_dict,
            "dynamic_keys": {
                "info_dict": new_keys_tracker.get("info_dict", []),
                "job_dict": new_keys_tracker.get("job_dict", [])
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        compute_and_store_ranking(user["uid"])

@web_app.get("/me")
async def get_my_profile(user: dict = Depends(get_current_user)):
    try:
        database = init_firebase()
        doc = database.collection("users").document(user["uid"]).get()
        if not doc.exists:
            raise HTTPException(status_code=404, detail="User data not found")
        return {"success": True, "data": doc.to_dict()}
    except ResourceExhausted:
        raise HTTPException(status_code=429, detail="Service usage limit exceeded")
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal Server Error")

@web_app.post("/match")
async def save_match(match_data: dict, user: dict = Depends(get_current_user)):
    try:
        job_id = match_data.get("job_id")
        score = match_data.get("score", 0)
        if not job_id:
            raise HTTPException(status_code=400, detail="job_id is required")
        database = init_firebase()
        database.collection("users").document(user["uid"]).collection("matches").document(job_id).set({
            "job_id": job_id,
            "score": score,
            "matched_at": firestore.SERVER_TIMESTAMP,
            "status": "saved"
        })
        return {"success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to save match")

@web_app.get("/matches")
async def get_matches(user: dict = Depends(get_current_user)):
    try:
        database = init_firebase()
        matches_ref = database.collection("users").document(user["uid"]).collection("matches").stream()
        matches = []
        for doc in matches_ref:
            match_data = doc.to_dict()
            job_id = match_data.get("job_id")
            saved_score = match_data.get("score", 0)
            job_doc = database.collection("resumes").document(job_id).get()
            if job_doc.exists:
                job_data = job_doc.to_dict()
                job_data = {key: job_data.get(key, "") for key in ["apply_options", "company_name", "description", "detected_extensions", "extensions", "job_highlights", "location", "title"]}
                job_data["id"] = job_id
                job_data["score"] = saved_score
                job_data["matched_at"] = match_data.get("matched_at")
                matches.append(job_data)
        return {"success": True, "matches": matches}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to fetch matches")

@web_app.delete("/match/{job_id}")
async def delete_match(job_id: str, user: dict = Depends(get_current_user)):
    try:
        database = init_firebase()
        database.collection("users").document(user["uid"]).collection("matches").document(job_id).delete()
        return {"success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to delete match")

@web_app.delete("/matches")
async def clear_all_matches(user: dict = Depends(get_current_user)):
    try:
        database = init_firebase()
        matches_ref = database.collection("users").document(user["uid"]).collection("matches").stream()
        for doc in matches_ref:
            doc.reference.delete()
        return {"success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to clear matches")

class ChatRequest(BaseModel):
    message: str
    job_id: str

CHAT_SYSTEM_PROMPT = """You are SwipeHire AI, a helpful career assistant integrated into a job discovery platform. 

You have access to:
1. The user's resume/profile (their skills, experience, education)
2. The current job they're viewing (title, company, description, requirements)

Your role is to:
- Answer questions about how well the user fits this specific job
- Highlight matching skills and experience
- Identify gaps and suggest how to address them
- Provide interview tips specific to this role
- Help craft tailored cover letter points
- Explain technical requirements the user may not understand

Be concise, helpful, and encouraging. Use bullet points when listing items.
Always reference specific details from both the resume and job posting."""

@web_app.post("/chat")
async def chat_with_job(request: ChatRequest, user: dict = Depends(get_current_user)):
    try:
        database = init_firebase()
        user_doc = database.collection("users").document(user["uid"]).get()
        if not user_doc.exists:
            raise HTTPException(status_code=404, detail="User data not found")
        user_data = user_doc.to_dict()
        info_dict = user_data.get("info_dict", {})
        job_dict = user_data.get("job_dict", {})
        job_doc = database.collection("resumes").document(request.job_id).get()
        if not job_doc.exists:
            raise HTTPException(status_code=404, detail="Job not found")
        job_data = job_doc.to_dict()
        resume_context = f"""
USER'S RESUME:
Name: {info_dict.get('name', 'Unknown')}
Email: {info_dict.get('email', 'N/A')}
Education: {json.dumps(info_dict.get('education', []), indent=2)}
Skills: {json.dumps(job_dict.get('technical_skills', []), indent=2)}
Experience: {json.dumps(job_dict.get('experience_summary', ''), indent=2)}
Projects: {json.dumps(job_dict.get('projects', []), indent=2)}
"""
        job_context = f"""
CURRENT JOB:
Title: {job_data.get('title', 'Unknown')}
Company: {job_data.get('company_name', 'Unknown')}
Location: {job_data.get('location', 'N/A')}
Description: {job_data.get('description', 'No description')[:3000]}
Requirements: {json.dumps(job_data.get('job_highlights', []), indent=2)}
Tags: {json.dumps(job_data.get('extensions', []), indent=2)}
"""
        full_prompt = f"""{CHAT_SYSTEM_PROMPT}

{resume_context}

{job_context}

USER QUESTION: {request.message}

Provide a helpful, concise response:"""

        async def generate():
            api_key = os.environ.get("GEMINI_API_KEY")
            client = genai.Client(api_key=api_key)
            try:
                response = client.models.generate_content_stream(
                    model="gemini-2.0-flash",
                    contents=full_prompt,
                )
                for chunk in response:
                    if chunk.text:
                        yield f"data: {json.dumps({'text': chunk.text})}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@web_app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.function(
    image=image,
    volumes={VOLUME_PATH: volume},
    secrets=[modal.Secret.from_name("gemini-secret")],
    timeout=300,
)
@modal.asgi_app()
def fastapi_app():
    init_firebase()
    return web_app