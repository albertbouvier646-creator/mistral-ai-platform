"""Verify Supabase-issued JWTs on incoming requests."""

import os
from fastapi import Header, HTTPException
from jose import jwt, JWTError

SUPABASE_JWT_SECRET = os.environ["SUPABASE_JWT_SECRET"]


def get_current_user_id(authorization: str = Header(...)) -> str:
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")

    token = authorization.removeprefix("Bearer ")
    try:
        payload = jwt.decode(token, SUPABASE_JWT_SECRET, algorithms=["HS256"], audience="authenticated")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    return payload["sub"]
