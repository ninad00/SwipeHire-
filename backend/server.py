import os
import json
from pydoc import doc
import tempfile
import numpy as np
import pytesseract
from pdf2image import convert_from_path
import traceback

from google import genai
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, Dict, List
import firebase_admin
from firebase_admin import credentials, firestore, auth
from google.api_core.exceptions import ResourceExhausted
from bandit import apply_feedback, rerank_jobs, rerank_jobs,build_features
from use_encoder import cross_encoder_rerank, cross_encoder_rerank, cross_encoder_rerank, load_cross_encoder
import asyncio



# Load environment variables
load_dotenv()


# Tesseract and Poppler paths
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
POPPLER_PATH = r"C:\poppler\Release-25.12.0-0\poppler-25.12.0\Library\bin"
SYSTEM_PROMPT_PATH = os.path.join(os.path.dirname(__file__),  "system_prompt.txt")

app = FastAPI(title="Resume Parser & Job Recommendation API")

cred_path = os.path.join(os.path.dirname(__file__), "job-swipe-1a26b-firebase-adminsdk-fbsvc-1e4a97ee2b.json")
cred = credentials.Certificate(cred_path)
firebase_admin.initialize_app(cred)
db = firestore.client()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173","http://localhost:3000", "http://127.0.0.1:5173", "http://localhost:8080", "http://localhost:8081", "http://127.0.0.1:8080", "http://127.0.0.1:8081"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

        # Return useful user info
        return {
            "uid": decoded_token.get("uid"),
            "email": decoded_token.get("email"),
            "name": decoded_token.get("name"),          # Google / provider name
       
        }

    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF using OCR."""
    try:
        images = convert_from_path(pdf_path, poppler_path=POPPLER_PATH)
        text_content = ""
        for image in images:
            text_content += pytesseract.image_to_string(image)
        return text_content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF extraction error: {str(e)}")

def parse_resume_with_llm(text_content: str) -> dict:
    """Parse resume text to JSON using Gemini LLM."""
    if not os.getenv("GEMINI_API_KEY"):
         raise HTTPException(status_code=500, detail="GEMINI_API_KEY not found in environment")
    
    if not os.path.exists(SYSTEM_PROMPT_PATH):
        raise HTTPException(status_code=500, detail="system_prompt.txt not found")

    with open(SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as f:
        system_prompt = f.read()

    client = genai.Client()
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
    """Create embedding using Gemini text-embedding-004."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not found in environment")
    
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
    """Load the job database with embeddings."""
    try:
        jobs_ref=db.collection("resumes").stream()
        jobs = []
        for doc in jobs_ref:
            data = doc.to_dict()
            data["id"] = doc.id   # keep document ID
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

    """Rank jobs by cosine similarity to user's job_dict."""
    job_text = json.dumps(job_dict, sort_keys=True)
    query_embedding = create_embedding(job_text)
    
    results = []

    for item in database:
        embedding = item.get("emb_new")
        if not embedding:
            continue  # skip jobs without embeddings

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

    # Sort by similarity score
    results.sort(key=lambda x: x["score"], reverse=True)
    print(f"Top job match score: {results[0]['score'] if results else 'N/A'}")
    print(f"Top company name: {results[0]['company'] if results else 'N/A'}")

    return results[:top_k], query_embedding.flatten()   


# def cross_encoder_rerank(
#     job_dict,
#     candidate_jobs,
#     cross_encoder,
#     top_k=20
# ):

#     resume_text = json.dumps(
#         job_dict,
#         sort_keys=True
#     )

#     pairs = []

#     for job in candidate_jobs:

#         job_text = (
#             f"{job.get('title','')} "
#             f"{job.get('description','')}"
#         )

#         pairs.append(
#             (
#                 resume_text,
#                 job_text
#             )
#         )

#     scores = cross_encoder.predict(
#         pairs
#     )

#     for job, score in zip(
#         candidate_jobs,
#         scores
#     ):
#         job["cross"] = float(score)

#     candidate_jobs.sort(
#         key=lambda x: x["cross"],
#         reverse=True
#     )

#     return candidate_jobs[:top_k]


def compute_and_store_ranking(uid):
    try:
        print(f"Starting ranking for user {uid}...")
        user_doc = db.collection("users").document(uid).get()
        if not user_doc.exists:
            print(f"User {uid} not found for ranking.")
            return

        user_data = user_doc.to_dict()
        job_dict = user_data.get("job_dict")
        
        if not job_dict:
            print(f"No job_dict found for user {uid}. Skipping ranking.")
            return

        database = load_job_database()
        print(f"Loaded {len(database)} jobs from database.")
        
        # Rank jobs and get objects with IDs and scores
        ranked_jobs = rank_jobs_by_similarity(job_dict, database, top_k=3810)
        print(f"Ranking complete. Top score: {ranked_jobs[0]['score'] if ranked_jobs else 'N/A'}")

        # De-duplicate by title: Keep only the first occurrence of each title
        seen_titles = set()
        unique_ranked_jobs = []
        for job in ranked_jobs:
            title = job.get("title", "").strip().lower()
            if title and title not in seen_titles:
                seen_titles.add(title)
                unique_ranked_jobs.append(job)
        
        print(f"De-duplicated: {len(ranked_jobs)} -> {len(unique_ranked_jobs)} unique jobs")

        # Store IDs and scores
        ranked_jobs_data = [{"id": job["id"], "score": job["score"]} for job in unique_ranked_jobs]

        db.collection("users").document(uid).update({
            "ranked_jobs": ranked_jobs_data,
             # Store full objects
            "ranking_updated_at": firestore.SERVER_TIMESTAMP
        })
        print(f"Successfully updated ranked_jobs for {uid}")


    except Exception as e:
        print(f"Error in compute_and_store_ranking: {str(e)}")
        import traceback
        traceback.print_exc()

# def full_enc_pipeline(uid):
#     try:
#         print(f"Starting ranking for user {uid}...")
#         user_doc = db.collection("users").document(uid).get()
#         if not user_doc.exists:
#             print(f"User {uid} not found for ranking.")
#             return

#         user_data = user_doc.to_dict()
#         job_dict = user_data.get("job_dict")
        
#         if not job_dict:
#             print(f"No job_dict found for user {uid}. Skipping ranking.")
#             return

#         database = load_job_database()
#         print(f"Loaded {len(database)} jobs from database.")
        
#         # Rank jobs and get objects with IDs and scores
#         ranked_jobs = rank_jobs_by_similarity(job_dict, database, top_k=3810)
#         print(f"Ranking complete. Top score: {ranked_jobs[0]['score'] if ranked_jobs else 'N/A'}")

#         # De-duplicate by title: Keep only the first occurrence of each title
#         seen_titles = set()
#         unique_ranked_jobs = []
#         for job in ranked_jobs:
#             title = job.get("title", "").strip().lower()
#             if title and title not in seen_titles:
#                 seen_titles.add(title)
#                 unique_ranked_jobs.append(job)
        
#         # cosine -> top100

#         top100 = ranked_jobs[:80]

#         cross_model, cross_tokenizer, cross_device = (
#     load_cross_encoder(
#         "best_model.pt",
#         "lora_adapter"
#     )
# )
#     # cross encoder -> top20
        
#         top20 = cross_encoder_rerank(
#             job_dict,
#             top100,
#             cross_model,
#             cross_tokenizer,
#             cross_device,
#             top_k=20
#         )

#         db.collection("users").document(uid).update({
#             "ranked_jobs": ranked_jobs_data, # Store full objects
#             "ranking_updated_at": firestore.SERVER_TIMESTAMP
#         })



#         # linucb rerank

#         resume_embedding = user_data[
#             "resume_embedding"
#         ]

#         final_jobs = rerank_jobs(
#             db=db,
#             uid=uid,
#             jobs=top20,
#             resume_embedding=resume_embedding
#         )

#         ranked_jobs_data = []

#         for item in final_jobs:

#             ranked_jobs_data.append(
#                 {
#                     "id": item["id"],

#                     # LinUCB score
#                     "score": item["score"],

#                     # needed later for feedback
#                     "features": item["features"]
#                 }
#             )

#         db.collection(
#             "users"
#         ).document(
#             uid
#         ).update(
#             {
#                 "ranked_jobs":
#                     ranked_jobs_data,

#                 "count": 0,

#                 "ranking_updated_at":
#                     firestore.SERVER_TIMESTAMP
#             }
#         )

#         print(
#             f"Stored {len(ranked_jobs_data)} jobs"
#         )


#     except Exception as e:
#         print(f"Error in compute_and_store_ranking: {str(e)}")
#         import traceback
#         traceback.print_exc()

def build_candidate_pool(uid):

    user_doc = (
        db.collection("users")
        .document(uid)
        .get()
    )

    user_data = user_doc.to_dict()

    job_dict = user_data["job_dict"]

    database = load_job_database()

    # cosine retrieval
    

    cosine_jobs, resume_embedding = rank_jobs_by_similarity(
        job_dict,
        database,
        top_k=3000
    )
    print(resume_embedding.shape)
    # dedupe

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

    # load once globally ideally
    

    # rerank

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

        # job_doc = (
        #     db.collection("resumes")
        #     .document(job["id"])
        #     .get()
        # )

        # job_emb = job_doc.to_dict()["emb_new"]

        candidate_pool.append(
            {
                "id": job["id"],
                "cross": job["cross"],
                "cosine": job["score"],
            }
        )

    db.collection("users") \
      .document(uid) \
      .update({
            "candidate_jobs":
                candidate_pool,

            "count": 0,
            "resume_embedding":resume_embedding.tolist(),
            "seen_jobs": [], # reset seen jobs

            "candidate_updated_at":
                firestore.SERVER_TIMESTAMP
      })
    
def rerank_for_login(uid):

    user_doc = (
        db.collection("users")
        .document(uid)
        .get()
    )

    user_data = user_doc.to_dict()

    candidate_jobs = (
        user_data.get(
            "candidate_jobs",
            []
        )
    )

    resume_embedding = (
        user_data[
            "resume_embedding"
        ]
    )

    jobs = []

    for item in candidate_jobs:

        job_doc = (
            db.collection("resumes")
            .document(item["id"])
            .get()
        )

        job_embedding = (
            job_doc.to_dict()["emb_new"]
        )

        jobs.append(
            {
                "id": item["id"],
                "embedding": job_embedding,
                "cosine": item["cosine"],
                "cross": item["cross"]
            }
        )

    final_jobs = rerank_jobs(
        db=db,
        uid=uid,
        jobs=jobs,
        resume_embedding=
            resume_embedding
    )

    ranked_jobs = []

    for item in final_jobs:

        ranked_jobs.append(
            {
                "id":
                    item["id"],

                "score":
                    item["score"],

                
            }
        )

    current_count = user_data.get("count", 0)

    db.collection("users") \
      .document(uid) \
      .update({
            "ranked_jobs":
                ranked_jobs,

            "count": current_count,

            "ranking_updated_at":
                firestore.SERVER_TIMESTAMP
      })

# def get_features(uid, job_id):

#     user_doc = (
#         db.collection("users")
#         .document(uid)
#         .get()
#     )

#     ranked_jobs = (
#         user_doc.to_dict()
#         .get("ranked_jobs", [])
#     )

#     for job in ranked_jobs:

#         if job["id"] == job_id:
#             return job["features"]

#     return None

def get_job_context(
    uid,
    job_id
):

    user_doc = (
        db.collection("users")
        .document(uid)
        .get()
    )

    user_data = user_doc.to_dict()

    resume_embedding = (
        user_data["resume_embedding"]
    )

    candidate_jobs = (
        user_data["candidate_jobs"]
    )

    for job in candidate_jobs:

        if job["id"] == job_id:
            job_doc = (
                db.collection("resumes")
                .document(job_id)
                .get()
            )

            job_embedding = (
                job_doc.to_dict()["emb_new"]
            )

            return {
                "resume_embedding":
                    resume_embedding,

                "job_embedding":
                    job_embedding,

                "cross":
                    job["cross"],

                "cosine":
                    job["cosine"]
            }

    return None

@app.get("/save-profile")
async def save_profile(user: dict = Depends(get_current_user)):
    """
    Save profile and get ranked job recommendations.
    
    Takes job_dict, creates embedding, and returns jobs ranked by similarity.
    """
    try:
        user_doc = db.collection("users").document(user["uid"]).get()
        if not user_doc.exists:
            print(f"User document not found for uid: {user['uid']}")
            raise HTTPException(status_code=404, detail="User data not found")
            
        rerank_for_login(user["uid"])

        user_doc = db.collection("users").document(user["uid"]).get()
        user_data = user_doc.to_dict()

        ranked_jobs_data = user_data.get("ranked_jobs", [])
        seen_jobs = set(
        user_data.get(
            "seen_jobs",
            []
        )
    )

        unseen_jobs = [
            job
            for job in ranked_jobs_data
            if job["id"] not in seen_jobs
        ]

        current_batch = unseen_jobs[:5]

        print(
        "ranked:",
        len(ranked_jobs_data),
        "seen:",
        len(seen_jobs),
        "unseen:",
        len(unseen_jobs),
    )

        jobs_to_send_details = []
        for item in current_batch:
            job_id = item["id"]
            score = item["score"]
            
            job_doc = db.collection("resumes").document(job_id).get()
            if job_doc.exists:
                job_data = job_doc.to_dict()
                # Filter fields
                job_data = {key: job_data.get(key, "") for key in ["apply_options", "company_name", "description", "detected_extensions", "extensions", "job_highlights", "location", "title"]}
                job_data["id"] = job_id
                job_data["score"] = score # Real score from ranking
                jobs_to_send_details.append(job_data)
            else:
                print(f"DEBUG: Job ID {job_id} not found in 'resumes' collection.")

        print("ranked_jobs_data =", len(ranked_jobs_data))
        print("seen_jobs =", len(seen_jobs))
        print("unseen_jobs =", len(unseen_jobs))
        print("current_batch =", len(current_batch))
        print("sample ranked jobs:", ranked_jobs_data[:3])
        print(f"DEBUG: Successfully fetched {len(jobs_to_send_details)} job details.")

        if jobs_to_send_details:

            shown_ids = [
                job["id"]
                for job in jobs_to_send_details
            ]

            db.collection("users") \
            .document(user["uid"]) \
            .update({
                    "seen_jobs":
                        firestore.ArrayUnion(
                            shown_ids
                        )
            })
             
        return {
            "success": True,
            "ranked_jobs": jobs_to_send_details,
            "total_jobs": len(ranked_jobs_data)
        }
        
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
   
@app.websocket("/ws/jobs")
async def jobs_ws(ws: WebSocket):
    await ws.accept()

    try:
        token = ws.query_params.get("token")

        if not token:
            await ws.close(code=1008)
            return

        decoded = auth.verify_id_token(token)
        uid = decoded["uid"]

        user_ref = (
            db.collection("users")
            .document(uid)
        )

        while True:

            data = json.loads(
                await ws.receive_text()
            )

            if data["type"] != "NEXT_JOB":
                continue

            user = (
                user_ref.get()
                .to_dict()
            )

            ranked_jobs_data = (
                user.get(
                    "ranked_jobs",
                    []
                )
            )

            seen_jobs = set(
                user.get(
                    "seen_jobs",
                    []
                )
            )

            unseen_jobs = [
                job
                for job in ranked_jobs_data
                if job["id"] not in seen_jobs
            ]

            if not unseen_jobs:

                rerank_for_login(uid)

                user = user_ref.get().to_dict()

                ranked_jobs_data = user.get(
                    "ranked_jobs",
                    []
                )

                seen_jobs = set(
                    user.get(
                        "seen_jobs",
                        []
                    )
                )

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

            doc = (
                db.collection("resumes")
                .document(job_id)
                .get()
            )

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

            await ws.send_json(
                {
                    "type": "JOB",
                    "job": job_data
                }
            )

            user_ref.update(
                {
                    "seen_jobs":
                        firestore.ArrayUnion(
                            [job_id]
                        )
                }
            )

    except WebSocketDisconnect:
        print("Client disconnected")

@app.post("/parse-resume")
async def parse_resume(file: UploadFile = File(...),user: dict = Depends(get_current_user)):
    """
    Upload a resume PDF and get parsed info_dict and job_dict.
    """
    # Validate file type
    if not file.filename.lower().endswith(('.pdf', '.docx')):
        raise HTTPException(status_code=400, detail="Only PDF and DOCX files are supported")
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        tmp_path = tmp_file.name
    
    try:
        # Extract text from PDF
        raw_text = extract_text_from_pdf(tmp_path)
        
        if not raw_text.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from PDF")
        
        # Parse with LLM
        parsed_data = parse_resume_with_llm(raw_text)
        
        # Get both info_dict and job_dict
        info_dict = parsed_data.get("info_dict", {})
        job_dict = parsed_data.get("job_dict", {})
        new_keys_tracker = parsed_data.get("new_keys_tracker", {})

        # FORCE OVERWRITE: Reset user data completely
        db.collection("users").document(user["uid"]).set(
            {
                "info_dict": info_dict,
                "job_dict": job_dict,
                "dynamic_keys": new_keys_tracker,
                "count": 0,
                "seen_jobs": [], # Clear old seen jobs
                "ranked_jobs": [], # Clear old rankings
                "updated_at": firestore.SERVER_TIMESTAMP,
            },
            merge=False, # ❌ CRITICAL: Overwrite previous data
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
        # Clean up temp file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

        
    

@app.get("/debug/ranked_jobs")
async def debug_ranked_jobs(user: dict = Depends(get_current_user)):
    """Debug endpoint: compute and return ranked jobs and a sample of the resumes DB.
    This is protected by the same auth used elsewhere and is intended for debugging only.
    """
    try:
        user_doc = db.collection("users").document(user["uid"]).get()
        if not user_doc.exists:
            raise HTTPException(status_code=404, detail="User data not found")

        job_dict = user_doc.to_dict().get("job_dict", {})
        database = load_job_database()

        # Compute ranking (in-memory, do not store)
        ranked = rank_jobs_by_similarity(job_dict, database, top_k=50)

        # Provide a sample mapping of doc ids -> titles from the DB to inspect whether doc ids are titles
        db_sample = [{"doc_id": d.get("id"), "title": d.get("title")} for d in database[:50]]

        return {
            "success": True,
            "ranked_jobs_sample": ranked[:20],
            "database_sample": db_sample
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.get("/me")
async def get_my_profile(user: dict = Depends(get_current_user)):
    try:
        doc = db.collection("users").document(user["uid"]).get()

        if not doc.exists:
            raise HTTPException(status_code=404, detail="User data not found")

        return {
            "success": True,
            "data": doc.to_dict()
        }
    except ResourceExhausted:
        raise HTTPException(
            status_code=429, 
            detail="Service usage limit exceeded. Please try again later or contact support."
        )
    except Exception as e:
        print(f"Error fetching profile: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.post("/match")
async def save_match(match_data: dict, user: dict = Depends(get_current_user)):
    """
    Save a job as a match for the user.
    """
    try:
        job_id = match_data.get("job_id")
        score = match_data.get("score", 0)  # Get score from request
        if not job_id:
            raise HTTPException(status_code=400, detail="job_id is required")

        # Save to 'matches' subcollection
        # Structure: users/{uid}/matches/{job_id}
        db.collection("users").document(user["uid"]).collection("matches").document(job_id).set({
            "job_id": job_id,
            "score": score,
            "matched_at": firestore.SERVER_TIMESTAMP,
            "status": "saved" 
        })
        ctx = get_job_context(
            user["uid"],
            job_id
        )

        if ctx:

            features = build_features(
                cosine_score=
                    ctx["cosine"],

                cross_score=
                    ctx["cross"],

                resume_embedding=
                    ctx["resume_embedding"],

                job_embedding=
                    ctx["job_embedding"]
            )

            apply_feedback(
                db=db,
                uid=user["uid"],
                features=features,
                reward=1
            )

        print(f"Received feedback from user {user['uid']}: reward={1} ")
        return {"success": True}
    except Exception as e:
        print(f"Error saving match: {e}")
        raise HTTPException(status_code=500, detail="Failed to save match")
    
@app.post("/not-match")
async def not_match(
    data: dict,
    user: dict = Depends(get_current_user)
):

    job_id = data["job_id"]

    ctx = get_job_context(
    user["uid"],
    job_id
)

    if ctx:

        features = build_features(
            cosine_score=
                ctx["cosine"],

            cross_score=
                ctx["cross"],

            resume_embedding=
                ctx["resume_embedding"],

            job_embedding=
                ctx["job_embedding"]
        )

        apply_feedback(
            db=db,
            uid=user["uid"],
            features=features,
            reward=-1
        )
        print(f"Received feedback from user {user['uid']}: reward={-1} ")

    return {"success": True}

@app.get("/matches")
async def get_matches(user: dict = Depends(get_current_user)):
    """
    Get all saved matches for the user.
    """
    try:
        matches_ref = db.collection("users").document(user["uid"]).collection("matches").stream()
        matches = []
        
        for doc in matches_ref:
            match_data = doc.to_dict()
            job_id = match_data.get("job_id")
            saved_score = match_data.get("score", 0)
            
            # Fetch job details
            job_doc = db.collection("resumes").document(job_id).get()
            if job_doc.exists:
                job_data = job_doc.to_dict()
                job_data = {key: job_data.get(key, "") for key in ["apply_options", "company_name", "description", "detected_extensions", "extensions", "job_highlights", "location", "title"]}
                job_data["id"] = job_id
                job_data["score"] = saved_score  # Include the saved score
                job_data["matched_at"] = match_data.get("matched_at")
                matches.append(job_data)
        
        return {"success": True, "matches": matches}
    except Exception as e:
        print(f"Error fetching matches: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch matches")


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


@app.post("/chat")
async def chat_with_job(request: ChatRequest, user: dict = Depends(get_current_user)):
    """
    Chat about a specific job using user's resume context.
    Returns streaming SSE response.
    """
    try:
        # Get user's resume data
        user_doc = db.collection("users").document(user["uid"]).get()
        if not user_doc.exists:
            raise HTTPException(status_code=404, detail="User data not found")
        
        user_data = user_doc.to_dict()
        info_dict = user_data.get("info_dict", {})
        job_dict = user_data.get("job_dict", {})
        
        # Get current job details
        job_doc = db.collection("resumes").document(request.job_id).get()
        if not job_doc.exists:
            raise HTTPException(status_code=404, detail="Job not found")
        
        job_data = job_doc.to_dict()
        
        # Build context
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

        # Create streaming generator
        async def generate():
            client = genai.Client()
            try:
                response = client.models.generate_content_stream(
                    model="gemma-4-31b-it",
                    contents=full_prompt,
                )
                
                for chunk in response:
                    if chunk.text:
                        # SSE format
                        yield f"data: {json.dumps({'text': chunk.text})}\n\n"
                
                yield "data: [DONE]\n\n"
                
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}



@app.post("/feedback")
async def feedback(
    data: dict,
    user: dict = Depends(get_current_user)
):

    apply_feedback(
        db=db,
        uid=user["uid"],
        features=data["features"],
        reward=data["reward"]
    )
    print(f"Received feedback from user {user['uid']}: reward={data['reward']} ")
    return {
        "success": True
    }

@app.websocket("/ws/apply")
async def apply_ws(ws: WebSocket):
    await ws.accept()
    
    match_ref = None
    job_id = None
    process = None
    
    try:
        # Get query parameters
        token = ws.query_params.get("token")
        job_id = ws.query_params.get("job_id")
        
        if not token or not job_id:
            await ws.send_json({"type": "error", "message": "Missing token or job_id"})
            await ws.close(code=1008)
            return
            
        # 1. Authenticate user
        try:
            decoded = auth.verify_id_token(token)
            uid = decoded["uid"]
        except Exception:
            await ws.send_json({"type": "error", "message": "Invalid token"})
            await ws.close(code=1008)
            return
            
        await ws.send_json({"type": "status", "message": "Authenticated. Fetching job details..."})
        
        # 2. Fetch Job Details from Firestore
        job_doc = db.collection("resumes").document(job_id).get()
        if not job_doc.exists:
            await ws.send_json({"type": "error", "message": "Job not found"})
            await ws.close()
            return
            
        job_data = job_doc.to_dict()
        title = job_data.get("title", "Job")
        company = job_data.get("company_name", "Company")
        
        # Find application link
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
        
        # 3. Update match status in Firestore to "applying"
        match_ref = db.collection("users").document(uid).collection("matches").document(job_id)
        match_ref.set({"status": "applying", "title": title, "company_name": company}, merge=True)
        
        # 4. Start the automation process as a separate OS process
        import subprocess
        import sys
        
        await ws.send_json({"type": "status", "message": "Launching browser automation process..."})
        
        python_exe = sys.executable
        script_path = os.path.join(os.path.dirname(__file__), "run_agent.py")
        
        env = os.environ.copy()
        
        process = subprocess.Popen(
            [python_exe, script_path, uid, job_id],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1, # Line buffered
            env=env
        )
        
        # Concurrently listen for client abort/close and read stdout of subprocess
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
            # Send sentinel when process stdout finishes
            loop.call_soon_threadsafe(ws_queue.put_nowait, None)

        def read_stderr():
            for line in iter(process.stderr.readline, ''):
                line = line.strip()
                if line:
                    print(f"[Agent Subprocess Error] {line}")
                    loop.call_soon_threadsafe(ws_queue.put_nowait, {"type": "status", "message": f"[System Log] {line}"})
                    
        # Launch reader threads
        import threading
        t_stdout = threading.Thread(target=read_stdout, daemon=True)
        t_stderr = threading.Thread(target=read_stderr, daemon=True)
        t_stdout.start()
        t_stderr.start()
        
        # Concurrently handle WebSocket client inputs (abort) and WebSocket writer
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
                    
                    # Persist status in Firestore
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
from fastapi.responses import FileResponse

@app.get("/live-screenshot/{uid}")
async def get_live_screenshot(uid: str):
    """
    Serve the live screenshot for the given user ID.
    """
    resumes_dir = os.path.join(os.path.dirname(__file__), "resumes")
    screenshot_path = os.path.join(resumes_dir, f"{uid}_live.png")
    if os.path.exists(screenshot_path):
        return FileResponse(screenshot_path, media_type="image/jpeg", headers={"Cache-Control": "no-store"})
    raise HTTPException(status_code=404, detail="Live screenshot not found")