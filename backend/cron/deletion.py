import os
import sys
import datetime
import firebase_admin
from firebase_admin import credentials, firestore
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

def main():
    print("="*60)
    print(f"Starting Job Deletion Cron at {datetime.datetime.now()}")
    print("="*60)

    now = datetime.datetime.now(datetime.timezone.utc)
    print(f"Current UTC time: {now}")

    # Query for expired documents where expiry_date is less than current time
    expired_ref = db.collection("resumes").where("expiry_date", "<", now)
    expired_docs = list(expired_ref.stream())

    print(f"Found {len(expired_docs)} expired job(s) in the database.")

    deleted_count = 0
    batch = db.batch()
    for doc in expired_docs:
        job_data = doc.to_dict()
        title = job_data.get("title", "Unknown Title")
        company = job_data.get("company_name", "Unknown Company")
        expiry = job_data.get("expiry_date")
        
        print(f"Deleting expired job: '{title}' at '{company}' (Expired at: {expiry})")
        batch.delete(doc.reference)
        deleted_count += 1

        # Commit batch every 100 deletions to stay safely below Firestore's 500 limit
        if deleted_count % 100 == 0:
            print("Committing batch delete...")
            batch.commit()
            batch = db.batch()

    if deleted_count % 100 != 0:
        print("Committing remaining deletions...")
        batch.commit()

    print("="*60)
    print(f"Cron Completed. Deleted {deleted_count} expired job(s).")
    print("="*60)

if __name__ == "__main__":
    main()
