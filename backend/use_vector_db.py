


import os

from typing import List, Dict, Tuple



import numpy as np

from supabase import create_client, Client





SUPABASE_URL = os.getenv("SUPABASE_URL")

SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")



supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)













def store_resume_embedding(uid: str, embedding: np.ndarray):




    supabase.table("user_embeddings").upsert({

        "user_id": uid,

        "embedding": embedding.tolist()

    }).execute()





def get_resume_embedding(uid: str) -> np.ndarray:




    result = (

        supabase.table("user_embeddings")

        .select("embedding")

        .eq("user_id", uid)

        .single()

        .execute()

    )



    return np.array(result.data["embedding"])













def store_job_embedding(job_id: str, embedding: np.ndarray):




    supabase.table("job_embeddings").upsert({

        "job_id": job_id,

        "embedding": embedding.tolist()

    }).execute()





def get_job_embedding(job_id: str) -> np.ndarray:




    result = (

        supabase.table("job_embeddings")

        .select("embedding")

        .eq("job_id", job_id)

        .single()

        .execute()

    )



    return np.array(result.data["embedding"])













def cosine_search(

    query_embedding: np.ndarray,

    top_k: int = 3000,

) -> List[Dict]:




    response = supabase.rpc(

        "match_job_embeddings",

        {

            "query_embedding": query_embedding.tolist(),

            "match_count": top_k,

        },

    ).execute()



    return response.data













def build_candidate_pool(

    query_embedding: np.ndarray,

    cross_encoder_fn,

):




    cosine_jobs = cosine_search(query_embedding, top_k=3000)



    seen_titles = set()

    unique_jobs = []



    for job in cosine_jobs:

        title = job["title"].strip().lower()



        if title in seen_titles:

            continue



        seen_titles.add(title)

        unique_jobs.append(job)



    top80 = cross_encoder_fn(unique_jobs[:80])



    return top80













def get_candidate_jobs(uid: str):



    return (

        supabase.table("candidate_jobs")

        .select("*")

        .eq("user_id", uid)

        .execute()

        .data

    )





def save_candidate_jobs(uid: str, jobs):



    rows = []



    for job in jobs:

        rows.append(

            {

                "user_id": uid,

                "job_id": job["id"],

                "cross": job["cross"],

                "cosine": job["score"],

            }

        )



    supabase.table("candidate_jobs").upsert(rows).execute()













def mark_seen(uid: str, job_id: str):



    supabase.table("seen_jobs").insert(

        {

            "user_id": uid,

            "job_id": job_id,

        }

    ).execute()





def get_seen_jobs(uid: str):



    return (

        supabase.table("seen_jobs")

        .select("job_id")

        .eq("user_id", uid)

        .execute()

        .data

    )
