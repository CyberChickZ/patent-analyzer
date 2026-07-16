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
