import os
import sys
import datetime
import time
from google import genai
import firebase_admin
from firebase_admin import credentials, firestore
import serpapi
from dotenv import load_dotenv

# 1. Load environment variables from backend/.env
base_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(base_dir, '../backend/.env')
load_dotenv(env_path)

# 2. Initialize Firebase Admin SDK using backend/firebase.json
cred_path = os.path.join(base_dir, '../backend/firebase.json')
if not firebase_admin._apps:
    print(f"Initializing Firebase with certificate: {cred_path}")
    cred = credentials.Certificate(cred_path)
    firebase_admin.initialize_app(cred)
db = firestore.client()

# 3. Initialize Gemini Client
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    print("CRITICAL ERROR: GEMINI_API_KEY environment variable is not set!")
    sys.exit(1)
gemini_client = genai.Client(api_key=gemini_api_key)

# 4. Initialize SerpApi Client
serp_api_key = "65472a1c2e85ac4105d408d125e13e4489282acbc3f1a2c47a504b5535ab1482"
serp_client = serpapi.Client(api_key=serp_api_key)

# 5. Target domains
DOMAINS = [
    "Machine Learning",
    "Data Science",
    "Healthcare",
    "Consulting",
    "Finance"
]

def fetch_jobs(query):
    print(f"Fetching jobs from SerpApi for query: '{query}'...")
    try:
        results = serp_client.search({
            "engine": "google_jobs",
            "location": "United States",
            "google_domain": "google.com",
            "hl": "en",
            "gl": "us",
            "q": query
        })
        jobs = results.get("jobs_results", [])
        print(f"Successfully fetched {len(jobs)} jobs for query: '{query}'")
        return jobs
    except Exception as e:
        print(f"Error fetching jobs for query '{query}': {e}")
        return []

def check_job_exists(job_id):
    if not job_id:
        return False
    # Check if this job_id is already in resumes collection
    docs = db.collection("resumes").where("job_id", "==", job_id).limit(1).get()
    return len(docs) > 0

from google.genai import types

def get_job_embedding(title, company_name, location, description):
    text_to_embed = f"Title: {title}\nCompany: {company_name}\nLocation: {location}\nDescription: {description}"
    retries = 3
    delay = 5
    for attempt in range(retries):
        try:
            response = gemini_client.models.embed_content(
                model="gemini-embedding-2",
                contents=text_to_embed,
                config=types.EmbedContentConfig(output_dimensionality=768)
            )
            # Add a small delay after a successful request to prevent hitting rate limits
            time.sleep(1.5)
            return response.embeddings[0].values
        except Exception as e:
            err_msg = str(e)
            if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg or "quota" in err_msg.lower():
                print(f"Rate limit hit for '{title}'. Retrying in {delay} seconds (Attempt {attempt+1}/{retries})...")
                time.sleep(delay)
                delay *= 2  # Exponential backoff
            else:
                print(f"Error generating embedding for job '{title}' at '{company_name}': {e}")
                return None
    print(f"Failed to generate embedding for job '{title}' after {retries} attempts.")
    return None

def main():
    print("="*60)
    print(f"Starting Job Aggregator Cron at {datetime.datetime.now()}")
    print("="*60)

    total_added = 0
    total_skipped = 0

    for domain in DOMAINS:
        jobs = fetch_jobs(domain)
        for job in jobs:
            job_id = job.get("job_id")
            title = job.get("title", "")
            company_name = job.get("company_name", "")
            location = job.get("location", "")
            description = job.get("description", "")

            if not job_id:
                print("Skipping job with no job_id")
                continue

            # De-duplicate check
            if check_job_exists(job_id):
                print(f"Job already exists (Skipping): '{title}' at '{company_name}' (ID: {job_id[:15]}...)")
                total_skipped += 1
                continue

            # Generate embedding
            print(f"Generating embedding for new job: '{title}' at '{company_name}'...")
            embedding = get_job_embedding(title, company_name, location, description)
            if not embedding:
                print("Skipping job due to embedding generation failure.")
                continue

            # Calculate dates
            now = datetime.datetime.now(datetime.timezone.utc)
            expiry = now + datetime.timedelta(days=30)  # 1 month expiry

            # Construct job document payload
            job_payload = {
                "title": title,
                "company_name": company_name,
                "location": location,
                "via": job.get("via", ""),
                "share_link": job.get("share_link", ""),
                "thumbnail": job.get("thumbnail", ""),
                "extensions": job.get("extensions", []),
                "detected_extensions": job.get("detected_extensions", {}),
                "description": description,
                "apply_options": job.get("apply_options", []),
                "job_id": job_id,
                "embedding": embedding,
                "added_at": now,
                "expiry_date": expiry
            }

            # Add to resumes collection
            try:
                db.collection("resumes").add(job_payload)
                print(f"SUCCESS: Added job '{title}' at '{company_name}' (Expiry: {expiry.strftime('%Y-%m-%d')})")
                total_added += 1
            except Exception as e:
                print(f"Error adding job to Firestore: {e}")

    print("="*60)
    print(f"Cron Completed. Added: {total_added}, Skipped: {total_skipped}")
    print("="*60)

if __name__ == "__main__":
    main()