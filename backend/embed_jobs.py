import os
import json
import time
import dotenv


import firebase_admin

from firebase_admin import (
    credentials,
    firestore,
)

from google import genai
dotenv.load_dotenv()

# =====================================================
# CONFIG
# =====================================================

FIREBASE_CRED_PATH = (
    "job-swipe-1a26b-firebase-adminsdk-fbsvc-1e4a97ee2b.json"
)

GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY"
)

COLLECTION_NAME = "resumes"

SLEEP_BETWEEN_REQUESTS = 0.2


# =====================================================
# FIREBASE
# =====================================================

cred = credentials.Certificate(
    FIREBASE_CRED_PATH
)

firebase_admin.initialize_app(
    cred
)


db = firestore.client()


# =====================================================
# GEMINI
# =====================================================




# =====================================================
# JOB -> TEXT
# =====================================================

def build_job_text(job):

    parts = []

    title = job.get("title")
    if title:
        parts.append(
            f"Job Title: {title}"
        )

    company = job.get(
        "company_name"
    )

    if company:
        parts.append(
            f"Company: {company}"
        )

    location = job.get(
        "location"
    )

    if location:
        parts.append(
            f"Location: {location}"
        )

    description = job.get(
        "description"
    )

    if description:
        parts.append(
            f"Description:\n{description}"
        )

    detected_extensions = job.get(
        "detected_extensions",
        {}
    )

    if isinstance(
        detected_extensions,
        dict
    ):

        extra_parts = []

        for key, value in (
            detected_extensions.items()
        ):

            if value is None:
                continue

            if isinstance(
                value,
                (dict, list)
            ):
                value = json.dumps(
                    value,
                    ensure_ascii=False
                )

            extra_parts.append(
                f"{key}: {value}"
            )

        if extra_parts:

            parts.append(
                "Additional Information:\n"
                + "\n".join(extra_parts)
            )

    return "\n\n".join(parts)


# =====================================================
# EMBEDDING
# =====================================================

def create_embedding(text):
    client = genai.Client(api_key=GEMINI_API_KEY)

    response = (
        client.models.embed_content(
            model="gemini-embedding-001",
            contents=text
        )
    )

    return (
        response.embeddings[0]
        .values
    )


# =====================================================
# MIGRATION
# =====================================================

def migrate_embeddings():

    docs = list(
        db.collection(
            COLLECTION_NAME
        ).stream()
    )

    total = len(docs)

    print(
        f"\nFound {total} jobs\n"
    )

    success = 0
    failed = 0

    for idx, doc in enumerate(
        docs,
        start=1
    ):

        try:

            data = doc.to_dict()

            # Skip already migrated jobs

            if data.get("emb_new"):

                print(
                    f"[{idx}/{total}] "
                    f"Skipping {doc.id}"
                )

                continue

            job_text = build_job_text(
                data
            )

            if not job_text.strip():

                print(
                    f"[{idx}/{total}] "
                    f"Empty job text: {doc.id}"
                )

                failed += 1
                continue

            embedding = create_embedding(
                job_text
            )

            doc.reference.update(
                {
                    "emb_new":
                        embedding,

                    
                }
            )

            success += 1

            print(
                f"[{idx}/{total}] "
                f"Updated {doc.id}"
            )

            time.sleep(
                SLEEP_BETWEEN_REQUESTS
            )

        except Exception as e:

            failed += 1

            print(
                f"\nFailed: {doc.id}"
            )

            print(e)
            print()

    print(
        "\n======================"
    )

    print(
        f"Success: {success}"
    )

    print(
        f"Failed: {failed}"
    )

    print(
        "Migration Complete"
    )

    print(
        "======================\n"
    )


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":

    migrate_embeddings()