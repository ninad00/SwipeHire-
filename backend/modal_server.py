import os
import sys
# Add current directory to Python path for container imports
sys.path.append(os.path.dirname(__file__))

import json
import tempfile
import numpy as np
import pytesseract
from pdf2image import convert_from_path
import asyncio

import modal
from google import genai
from google.genai import types
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import Depends, FastAPI, Header, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, Dict, List
import firebase_admin
from firebase_admin import credentials, firestore, auth
from google.api_core.exceptions import ResourceExhausted

from bandit import apply_feedback, rerank_jobs, build_features
from use_encoder import cross_encoder_rerank, load_cross_encoder

app = modal.App("tfj-backend")

volume = modal.Volume.from_name("tfj-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "tesseract-ocr",
        "poppler-utils",
        "libgl1-mesa-glx",
        "libglib2.0-0"
    )
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
        "torch",
        "transformers",
        "peft",
        "browser-use",
        "playwright",  # add this
    )
    .run_commands(
        "python -m playwright install --with-deps chromium"
    )
    .add_local_dir(
        os.path.dirname(__file__),
        remote_path="/root",
        ignore=["*.pt", "**/__pycache__"]
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

# Symlink best_model.pt and firebase.json from the persistent tfj-data volume if present
for filename in ["best_model.pt", "firebase.json"]:
    if os.path.exists(f"/data/{filename}") and not os.path.exists(filename):
        try:
            os.symlink(f"/data/{filename}", filename)
        except Exception as e:
            print(f"Failed to symlink {filename} from volume: {e}")

cross_model, cross_tokenizer, cross_device = (
    load_cross_encoder(
        "best_model.pt",
        "lora_adapter"
    )
)

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
            model="gemma-4-31b-it", 
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
            model="gemini-embedding-001",
            contents=text
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

def rank_jobs_by_similarity(
    job_dict: dict,
    database: List[dict],
    top_k: int = 50):
    
    job_text = json.dumps(job_dict, sort_keys=True)
    query_embedding = create_embedding(job_text)
    
    results = []
    for item in database:
        embedding = item.get("emb_new")
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
    print(f"Top job match score: {results[0]['score'] if results else 'N/A'}")
    print(f"Top company name: {results[0]['company'] if results else 'N/A'}")

    return results[:top_k], query_embedding.flatten()

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
        ranked_jobs, _ = rank_jobs_by_similarity(job_dict, job_database, top_k=3810)
        
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

def build_candidate_pool(uid):
    database = init_firebase()
    user_doc = (
        database.collection("users")
        .document(uid)
        .get()
    )
    user_data = user_doc.to_dict()
    job_dict = user_data["job_dict"]
    job_database = load_job_database()

    cosine_jobs, resume_embedding = rank_jobs_by_similarity(
        job_dict,
        job_database,
        top_k=3000
    )
    print(resume_embedding.shape)

    seen = set()
    unique_jobs = []
    for job in cosine_jobs:
        title = (
            job.get("title", "")
            .strip()
            .lower()
        )
        if title in seen:
            continue
        seen.add(title)
        unique_jobs.append(job)

    top500 = cross_encoder_rerank(
        job_dict,
        unique_jobs[:80],
        cross_model,
        cross_tokenizer,
        cross_device,
        top_k=80
    )

    candidate_pool = []
    for job in top500:
        candidate_pool.append(
            {
                "id": job["id"],
                "cross": job["cross"],
                "cosine": job["score"],
            }
        )

    database.collection("users") \
      .document(uid) \
      .update({
            "candidate_jobs": candidate_pool,
            "count": 0,
            "resume_embedding": resume_embedding.tolist(),
            "seen_jobs": [],
            "candidate_updated_at": firestore.SERVER_TIMESTAMP
      })

def rerank_for_login(uid):
    database = init_firebase()
    user_doc = (
        database.collection("users")
        .document(uid)
        .get()
    )
    user_data = user_doc.to_dict()
    candidate_jobs = user_data.get("candidate_jobs", [])
    resume_embedding = user_data["resume_embedding"]

    jobs = []
    for item in candidate_jobs:
        job_doc = (
            database.collection("resumes")
            .document(item["id"])
            .get()
        )
        job_embedding = job_doc.to_dict()["emb_new"]
        jobs.append(
            {
                "id": item["id"],
                "embedding": job_embedding,
                "cosine": item["cosine"],
                "cross": item["cross"]
            }
        )

    final_jobs = rerank_jobs(
        db=database,
        uid=uid,
        jobs=jobs,
        resume_embedding=resume_embedding
    )

    ranked_jobs = []
    for item in final_jobs:
        ranked_jobs.append(
            {
                "id": item["id"],
                "score": item["score"],
            }
        )

    current_count = user_data.get("count", 0)
    database.collection("users") \
      .document(uid) \
      .update({
            "ranked_jobs": ranked_jobs,
            "count": current_count,
            "ranking_updated_at": firestore.SERVER_TIMESTAMP
      })

def get_job_context(uid, job_id):
    database = init_firebase()
    user_doc = (
        database.collection("users")
        .document(uid)
        .get()
    )
    user_data = user_doc.to_dict()
    resume_embedding = user_data["resume_embedding"]
    candidate_jobs = user_data["candidate_jobs"]

    for job in candidate_jobs:
        if job["id"] == job_id:
            job_doc = (
                database.collection("resumes")
                .document(job_id)
                .get()
            )
            job_embedding = job_doc.to_dict()["emb_new"]
            return {
                "resume_embedding": resume_embedding,
                "job_embedding": job_embedding,
                "cross": job["cross"],
                "cosine": job["cosine"]
            }
    return None

@web_app.get("/save-profile")
async def save_profile(user: dict = Depends(get_current_user)):
    try:
        database = init_firebase()
        user_doc = database.collection("users").document(user["uid"]).get()
        if not user_doc.exists:
            print(f"User document not found for uid: {user['uid']}")
            raise HTTPException(status_code=404, detail="User data not found")
            
        rerank_for_login(user["uid"])

        user_doc = database.collection("users").document(user["uid"]).get()
        user_data = user_doc.to_dict()

        ranked_jobs_data = user_data.get("ranked_jobs", [])
        seen_jobs = set(user_data.get("seen_jobs", []))

        unseen_jobs = [
            job
            for job in ranked_jobs_data
            if job["id"] not in seen_jobs
        ]

        current_batch = unseen_jobs[:5]

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
            else:
                print(f"DEBUG: Job ID {job_id} not found in 'resumes' collection.")
 
        print(f"DEBUG: Successfully fetched {len(jobs_to_send_details)} job details.")

        if jobs_to_send_details:
            shown_ids = [job["id"] for job in jobs_to_send_details]
            database.collection("users") \
            .document(user["uid"]) \
            .update({
                    "seen_jobs": firestore.ArrayUnion(shown_ids)
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

        while True:
            data = json.loads(await ws.receive_text())
            if data["type"] != "NEXT_JOB":
                continue

            user = user_ref.get().to_dict()
            ranked_jobs_data = user.get("ranked_jobs", [])
            seen_jobs = set(user.get("seen_jobs", []))

            unseen_jobs = [
                job
                for job in ranked_jobs_data
                if job["id"] not in seen_jobs
            ]

            if not unseen_jobs:
                rerank_for_login(uid)
                user = user_ref.get().to_dict()
                ranked_jobs_data = user.get("ranked_jobs", [])
                seen_jobs = set(user.get("seen_jobs", []))
                unseen_jobs = [
                    job
                    for job in ranked_jobs_data
                    if job["id"] not in seen_jobs
                ]
                if not unseen_jobs:
                    await ws.send_json({"type": "END"})
                    continue

            item = unseen_jobs[0]
            job_id = item["id"]
            score = item["score"]

            doc = database.collection("resumes").document(job_id).get()
            if not doc.exists:
                continue

            job_data = doc.to_dict()
            job_data = {
                key: job_data.get(key, "")
                for key in [
                    "apply_options",
                    "company_name",
                    "description",
                    "detected_extensions",
                    "extensions",
                    "job_highlights",
                    "location",
                    "title",
                ]
            }
            job_data["id"] = job_id
            job_data["score"] = score

            await ws.send_json({"type": "JOB", "job": job_data})
            user_ref.update({
                "seen_jobs": firestore.ArrayUnion([job_id])
            })
    except WebSocketDisconnect:
        print("Client disconnected")

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
        
        # Reset user data completely
        database.collection("users").document(user["uid"]).set(
            {
                "info_dict": info_dict,
                "job_dict": job_dict,
                "dynamic_keys": new_keys_tracker,
                "count": 0,
                "seen_jobs": [],
                "ranked_jobs": [],
                "updated_at": firestore.SERVER_TIMESTAMP,
            },
            merge=False,
        )
        build_candidate_pool(user["uid"])
        rerank_for_login(user["uid"])
        
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

@web_app.get("/debug/ranked_jobs")
async def debug_ranked_jobs(user: dict = Depends(get_current_user)):
    try:
        database = init_firebase()
        user_doc = database.collection("users").document(user["uid"]).get()
        if not user_doc.exists:
            raise HTTPException(status_code=404, detail="User data not found")

        job_dict = user_doc.to_dict().get("job_dict", {})
        job_database = load_job_database()

        # Compute ranking (in-memory, do not store)
        ranked, _ = rank_jobs_by_similarity(job_dict, job_database, top_k=50)

        db_sample = [{"doc_id": d.get("id"), "title": d.get("title")} for d in job_database[:50]]

        return {
            "success": True,
            "ranked_jobs_sample": ranked[:20],
            "database_sample": db_sample
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@web_app.get("/me")
async def get_my_profile(user: dict = Depends(get_current_user)):
    try:
        database = init_firebase()
        doc = database.collection("users").document(user["uid"]).get()
        if not doc.exists:
            raise HTTPException(status_code=404, detail="User data not found")
        return {"success": True, "data": doc.to_dict()}
    except ResourceExhausted:
        raise HTTPException(
            status_code=429, 
            detail="Service usage limit exceeded. Please try again later or contact support."
        )
    except Exception as e:
        print(f"Error fetching profile: {e}")
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
        ctx = get_job_context(user["uid"], job_id)
        if ctx:
            features = build_features(
                cosine_score=ctx["cosine"],
                cross_score=ctx["cross"],
                resume_embedding=ctx["resume_embedding"],
                job_embedding=ctx["job_embedding"]
            )
            apply_feedback(
                db=database,
                uid=user["uid"],
                features=features,
                reward=1
            )
        print(f"Received feedback from user {user['uid']}: reward=1")
        return {"success": True}
    except Exception as e:
        print(f"Error saving match: {e}")
        raise HTTPException(status_code=500, detail="Failed to save match")

@web_app.post("/not-match")
async def not_match(data: dict, user: dict = Depends(get_current_user)):
    job_id = data["job_id"]
    database = init_firebase()
    ctx = get_job_context(user["uid"], job_id)
    if ctx:
        features = build_features(
            cosine_score=ctx["cosine"],
            cross_score=ctx["cross"],
            resume_embedding=ctx["resume_embedding"],
            job_embedding=ctx["job_embedding"]
        )
        apply_feedback(
            db=database,
            uid=user["uid"],
            features=features,
            reward=-1
        )
        print(f"Received feedback from user {user['uid']}: reward=-1")
    return {"success": True}

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
                    model="gemma-4-31b-it",
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

@web_app.post("/feedback")
async def feedback(data: dict, user: dict = Depends(get_current_user)):
    database = init_firebase()
    apply_feedback(
        db=database,
        uid=user["uid"],
        features=data["features"],
        reward=data["reward"]
    )
    print(f"Received feedback from user {user['uid']}: reward={data['reward']}")
    return {"success": True}

@web_app.websocket("/ws/apply")
async def apply_ws(ws: WebSocket):
    await ws.accept()
    
    match_ref = None
    job_id = None
    process = None
    
    try:
        token = ws.query_params.get("token")
        job_id = ws.query_params.get("job_id")
        
        if not token or not job_id:
            await ws.send_json({"type": "error", "message": "Missing token or job_id"})
            await ws.close(code=1008)
            return
            
        try:
            decoded = auth.verify_id_token(token)
            uid = decoded["uid"]
        except Exception:
            await ws.send_json({"type": "error", "message": "Invalid token"})
            await ws.close(code=1008)
            return
            
        await ws.send_json({"type": "status", "message": "Authenticated. Fetching job details..."})
        
        database = init_firebase()
        job_doc = database.collection("resumes").document(job_id).get()
        if not job_doc.exists:
            await ws.send_json({"type": "error", "message": "Job not found"})
            await ws.close()
            return
            
        job_data = job_doc.to_dict()
        title = job_data.get("title", "Job")
        company = job_data.get("company_name", "Company")
        
        apply_link = None
        apply_options = job_data.get("apply_options", [])
        if apply_options and isinstance(apply_options, list):
            apply_link = apply_options[0].get("link")
        if not apply_link:
            apply_link = job_data.get("share_link")
            
        if not apply_link:
            await ws.send_json({"type": "error", "message": "No application link found for this job"})
            await ws.close()
            return
            
        await ws.send_json({"type": "status", "message": f"Found application link: {apply_link}. Fetching user details..."})
        
        match_ref = database.collection("users").document(uid).collection("matches").document(job_id)
        match_ref.set({"status": "applying", "title": title, "company_name": company}, merge=True)
        
        await ws.send_json({"type": "status", "message": "Launching browser automation process..."})
        
        import subprocess
        import sys
        import asyncio
        import threading
        
        python_exe = sys.executable
        script_path = os.path.join(os.path.dirname(__file__), "run_agent.py")
        
        env = os.environ.copy()
        
        process = subprocess.Popen(
            [python_exe, script_path, uid, job_id],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env
        )
        
        loop = asyncio.get_running_loop()
        ws_queue = asyncio.Queue()
        
        def read_stdout():
            for line in iter(process.stdout.readline, ''):
                line = line.strip()
                if line:
                    try:
                        msg = json.loads(line)
                        loop.call_soon_threadsafe(ws_queue.put_nowait, msg)
                    except json.JSONDecodeError:
                        loop.call_soon_threadsafe(ws_queue.put_nowait, {"type": "status", "message": line})
            loop.call_soon_threadsafe(ws_queue.put_nowait, None)

        def read_stderr():
            for line in iter(process.stderr.readline, ''):
                line = line.strip()
                if line:
                    print(f"[Agent Subprocess Error] {line}")
                    loop.call_soon_threadsafe(ws_queue.put_nowait, {"type": "status", "message": f"[System Log] {line}"})
                    
        t_stdout = threading.Thread(target=read_stdout, daemon=True)
        t_stderr = threading.Thread(target=read_stderr, daemon=True)
        t_stdout.start()
        t_stderr.start()
        
        async def client_reader():
            try:
                while process.poll() is None:
                    text_data = await ws.receive_text()
                    data = json.loads(text_data)
                    if data.get("type") == "abort":
                        print("Received abort command. Terminating agent subprocess...")
                        process.kill()
                        if match_ref:
                            match_ref.update({
                                "status": "failed",
                                "failure_reason": "Aborted by user"
                            })
                        break
            except WebSocketDisconnect:
                print("Client disconnected. Terminating agent subprocess...")
                process.kill()
            except Exception as e:
                print(f"Error in client_reader: {e}")
                process.kill()

        async def client_writer():
            try:
                while True:
                    msg = await ws_queue.get()
                    if msg is None:
                        break
                    
                    await ws.send_json(msg)
                    
                    if msg.get("type") == "success":
                        if match_ref:
                            match_ref.update({
                                "status": "applied",
                                "applied_at": firestore.SERVER_TIMESTAMP,
                                "application_result": msg.get("result")
                            })
                    elif msg.get("type") == "error":
                        if match_ref:
                            match_ref.update({
                                "status": "failed",
                                "failure_reason": msg.get("message")
                            })
            except Exception as e:
                print(f"Error in client_writer: {e}")
                
        await asyncio.gather(client_reader(), client_writer(), return_exceptions=True)
        
    except WebSocketDisconnect:
        print(f"WS client disconnected during application process for job {job_id}")
        if process:
            process.kill()
        try:
            if match_ref:
                doc = match_ref.get()
                if doc.exists and doc.to_dict().get("status") == "applying":
                    match_ref.update({"status": "saved"})
        except Exception:
            pass
            
    except Exception as e:
        print(f"Error in apply_ws: {e}")
        if process:
            process.kill()
        try:
            await ws.send_json({"type": "error", "message": f"Application failed: {str(e)}"})
            if match_ref:
                match_ref.update({
                    "status": "failed",
                    "failure_reason": str(e)
                })
        except Exception:
            pass
    finally:
        try:
            await ws.close()
        except Exception:
            pass

@web_app.get("/live-screenshot/{uid}")
async def get_live_screenshot(uid: str):
    resumes_dir = os.path.join(os.path.dirname(__file__), "resumes")
    screenshot_path = os.path.join(resumes_dir, f"{uid}_live.png")
    if os.path.exists(screenshot_path):
        return FileResponse(screenshot_path, media_type="image/jpeg", headers={"Cache-Control": "no-store"})
    raise HTTPException(status_code=404, detail="Live screenshot not found")

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