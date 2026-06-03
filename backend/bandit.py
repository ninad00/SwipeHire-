import numpy as np
from firebase_admin import firestore

DIM = 3074


class DiagLinUCB:

    def __init__(
        self,
        alpha=0.5,
        A_diag=None,
        b=None
    ):

        self.alpha = alpha

        self.A_diag = (
            np.ones(DIM, dtype=np.float32)
            if A_diag is None
            else np.array(A_diag, dtype=np.float32)
        )

        self.b = (
            np.zeros(DIM, dtype=np.float32)
            if b is None
            else np.array(b, dtype=np.float32)
        )

    @property
    def theta(self):
        return self.b / self.A_diag

    def score(self, x):

        exploit = float(
            self.theta @ x
        )

        explore = (
            self.alpha *
            np.sqrt(
                np.sum(
                    (x * x) /
                    self.A_diag
                )
            )
        )

        return exploit + explore

    def update(
        self,
        x,
        reward
    ):

        self.A_diag += x * x
        self.b += reward * x


def load_user_model(
    db,
    uid
):

    doc = (
        db.collection("user_bandits")
        .document(uid)
        .get()
    )

    if not doc.exists:
        return DiagLinUCB()

    data = doc.to_dict()

    return DiagLinUCB(
        A_diag=data["a_diag"],
        b=data["b"]
    )


def save_user_model(
    db,
    uid,
    model
):

    (
        db.collection("user_bandits")
        .document(uid)
        .set(
            {
                "a_diag": model.A_diag.tolist(),
                "b": model.b.tolist(),
                "updated_at":
                    firestore.SERVER_TIMESTAMP
            }
        )
    )


def build_features(
    cosine_score,
    cross_score,
    resume_embedding,
    job_embedding
):

    interaction = (
        np.array(
            resume_embedding,
            dtype=np.float32
        )
        *
        np.array(
            job_embedding,
            dtype=np.float32
        )
    )

    return np.array(
        [
            cross_score,
            cosine_score,
            *interaction
        ],
        dtype=np.float32
    )


def rerank_jobs(
    db,
    uid,
    jobs,
    resume_embedding
):

    model = load_user_model(
        db,
        uid
    )

    scored = []

    for job in jobs:

        x = build_features(
            cosine_score=job["cosine"],
            cross_score=job["cross"],
            resume_embedding=resume_embedding,
            job_embedding=job["embedding"]
        )

        bandit_score = model.score(x)

        final_score=(0.8*job["cross"]+0.2*np.tanh(bandit_score))

        scored.append(
            {
                "id": job["id"],
                "score": float(
                    final_score
                ),
                "features":
                    x.tolist(),
                "job":
                    job
            }
        )

    scored.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return scored


def apply_feedback(
    db,
    uid,
    features,
    reward
):

    model = load_user_model(
        db,
        uid
    )

    x = np.array(
        features,
        dtype=np.float32
    )

    model.update(
        x,
        reward
    )

    save_user_model(
        db,
        uid,
        model
    )


REWARDS = {
    "skip": 0.0,
    "save": 1.0,
    "apply": 3.0
}