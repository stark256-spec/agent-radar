"""
JWT-based auth for the AgentRadar API.

In production: integrate Azure AD MSAL (client_credentials or authorization_code flow).
For local dev / self-hosted: use the built-in username/password JWT flow below.

Roles:
  admin   — full read/write access
  viewer  — read-only access to metrics, anomalies, violations
  auditor — read-only + export compliance reports
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

SECRET_KEY = os.getenv("AGENT_RADAR_SECRET", "change-me-in-production-use-32-char-secret")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 480  # 8 hours

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")

# In production: load from DB. For local dev, use env vars.
_DEMO_USERS = {
    "admin": {
        "hashed_password": pwd_context.hash(os.getenv("ADMIN_PASSWORD", "admin123")),
        "role": "admin",
    },
    "viewer": {
        "hashed_password": pwd_context.hash(os.getenv("VIEWER_PASSWORD", "viewer123")),
        "role": "viewer",
    },
}


class TokenData(BaseModel):
    username: str
    role: str


class Token(BaseModel):
    access_token: str
    token_type: str
    role: str


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def authenticate_user(username: str, password: str) -> dict | None:
    user = _DEMO_USERS.get(username)
    if not user or not verify_password(password, user["hashed_password"]):
        return None
    return {"username": username, **user}


def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=15))
    to_encode["exp"] = expire
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


async def get_current_user(token: str = Depends(oauth2_scheme)) -> TokenData:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub", "")
        role: str = payload.get("role", "viewer")
        if not username:
            raise credentials_exception
        return TokenData(username=username, role=role)
    except JWTError:
        raise credentials_exception


async def require_admin(current_user: TokenData = Depends(get_current_user)) -> TokenData:
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin role required")
    return current_user


async def require_viewer(current_user: TokenData = Depends(get_current_user)) -> TokenData:
    if current_user.role not in ("admin", "viewer", "auditor"):
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    return current_user
