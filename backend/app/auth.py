import os

import firebase_admin
from fastapi import HTTPException, Request
from firebase_admin import auth, credentials

_app = None


def _init():
    global _app
    if _app is not None:
        return
    cred_path = os.getenv("FIREBASE_CREDENTIALS")
    if cred_path:
        cred = credentials.Certificate(cred_path)
        _app = firebase_admin.initialize_app(cred)
    else:
        _app = firebase_admin.initialize_app()


def verify_token(token: str) -> dict:
    _init()
    try:
        decoded = auth.verify_id_token(token)
    except Exception as e:
        raise HTTPException(401, f"Invalid token: {e}")
    email = decoded.get("email", "")
    if not email.endswith("@oregonstate.edu"):
        raise HTTPException(403, "Only @oregonstate.edu accounts are allowed")
    return decoded


async def require_auth(request: Request) -> dict:
    if os.getenv("AUTH_DISABLED", "").lower() in ("1", "true"):
        return {"email": "dev@oregonstate.edu", "developer": True}
    token = request.headers.get("X-Firebase-Token")
    if not token:
        raise HTTPException(401, "Missing authentication token")
    return verify_token(token)


async def optional_auth(request: Request) -> dict | None:
    """A token if one came, otherwise None — the route decides.

    This is a deliberate hole and it belongs to exactly three routes:
    /status/{id}, /report/{id} and /results/{id}. A report is shared by sending
    somebody its URL, so requiring a login there would break the only way
    anybody outside the team reads one (see leader_deploy.md §3.2, where the
    report URL is handed out exactly like that).

    Everything that is about the DEPLOYMENT rather than one shared job now uses
    require_auth: /api/quota exposed what is left on every external key, and
    /api/jobs/{id}/usage and /funnel expose one job's spend and its whole search
    funnel. Those were never meant to be public; they were `optional_auth` only
    because it was the default in that block. Measured before the change: an
    unauthenticated /api/quota returned the full quota JSON and
    /api/status/99a35c00 the full job record (cloud_smoke.md §4, 2026-09-19).

    A job id is 8 hex characters, so "public with the link" is the honest way to
    describe these three, not "private".
    """
    token = request.headers.get("X-Firebase-Token")
    if not token:
        return None
    try:
        return verify_token(token)
    except HTTPException:
        return None


def is_developer(decoded: dict) -> bool:
    return bool(decoded.get("developer"))


def set_developer(email: str, is_dev: bool = True):
    _init()
    user = auth.get_user_by_email(email)
    claims = {"developer": True} if is_dev else {}
    auth.set_custom_user_claims(user.uid, claims)
    return user.uid
