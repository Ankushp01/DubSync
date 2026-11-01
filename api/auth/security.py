"""Security primitives: Argon2id password hashing, JWT tokens, and secure cookies."""
from __future__ import annotations

import hashlib
import os
import re
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

import jwt
from argon2 import PasswordHasher, Type
from argon2.exceptions import InvalidHashError, VerifyMismatchError
from fastapi import Response

# Password hasher configuration (Argon2id)
_ph = PasswordHasher(
    time_cost=2,
    memory_cost=65536,  # 64 MB
    parallelism=1,
    hash_len=32,
    type=Type.ID,
)

# JWT configuration
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "dubsync-jwt-production-secret-key-must-be-at-least-32-chars-long")
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "15"))
REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "14"))

# Cookie settings
COOKIE_ACCESS_NAME = "access_token"
COOKIE_REFRESH_NAME = "refresh_token"
COOKIE_CSRF_NAME = "csrf_token"

COOKIE_SECURE = os.getenv("COOKIE_SECURE", "false").lower() == "true"
COOKIE_SAMESITE = os.getenv("COOKIE_SAMESITE", "lax")
COOKIE_DOMAIN = os.getenv("COOKIE_DOMAIN", None)


def hash_password(password: str) -> str:
    """Hash password using Argon2id."""
    return _ph.hash(password)


def verify_password(hash_str: str, password: str) -> bool:
    """Verify password against stored Argon2id hash."""
    if not hash_str:
        return False
    try:
        return _ph.verify(hash_str, password)
    except (VerifyMismatchError, InvalidHashError):
        return False
    except Exception:
        return False


def validate_password_strength(password: str) -> tuple[bool, Optional[str]]:
    """Enforce reasonable password strength: minimum 8 characters, with letters and numbers."""
    if len(password) < 8:
        return False, "Password must be at least 8 characters long."
    if len(password) > 128:
        return False, "Password cannot exceed 128 characters."
    if not re.search(r"[A-Za-z]", password):
        return False, "Password must contain at least one letter."
    if not re.search(r"[0-9\W_]", password):
        return False, "Password must contain at least one number or special character."
    return True, None


def generate_random_token(nbytes: int = 32) -> str:
    """Generate cryptographically secure random url-safe token."""
    return secrets.token_urlsafe(nbytes)


def generate_csrf_token(nbytes: int = 32) -> str:
    """Generate cryptographically secure CSRF token."""
    return secrets.token_urlsafe(nbytes)


def hash_token(token: str) -> str:
    """Hash a token using SHA-256 for secure storage."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def create_access_token(
    user_id: Optional[str] = None,
    email: Optional[str] = None,
    subject: Optional[str] = None,
    claims: Optional[Dict[str, Any]] = None,
    role: str = "user",
    expires_delta: Optional[timedelta] = None,
) -> str:
    """Create short-lived JWT access token supporting both keyword calling styles."""
    sub = user_id or subject
    if not sub:
        raise ValueError("user_id or subject is required")
    em = email or (claims.get("email") if claims else None) or ""
    r = (claims.get("role") if claims else None) or role
    now = datetime.now(timezone.utc)
    exp = now + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    payload: Dict[str, Any] = {
        "sub": sub,
        "email": em,
        "role": r,
        "type": "access",
        "iat": int(now.timestamp()),
        "exp": int(exp.timestamp()),
    }
    if claims:
        for k, v in claims.items():
            if k not in payload:
                payload[k] = v
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def decode_access_token(token: str) -> Dict[str, Any]:
    """Decode and validate access token signature and expiration."""
    try:
        payload = jwt.decode(
            token,
            JWT_SECRET_KEY,
            algorithms=[JWT_ALGORITHM],
            options={"require": ["sub", "exp", "type"]},
        )
        if payload.get("type") != "access":
            raise jwt.InvalidTokenError("Invalid token type")
        return payload
    except jwt.ExpiredSignatureError:
        raise jwt.ExpiredSignatureError("Access token has expired")
    except Exception as e:
        raise jwt.InvalidTokenError(f"Invalid token: {e}")


def set_auth_cookies(
    response: Response,
    access_token: str,
    refresh_token: str,
    csrf_token: Optional[str] = None,
):
    """Set HttpOnly cookies for access and refresh tokens, and readable CSRF cookie."""
    access_max_age = ACCESS_TOKEN_EXPIRE_MINUTES * 60
    refresh_max_age = REFRESH_TOKEN_EXPIRE_DAYS * 24 * 3600

    # 1. HttpOnly Access Token
    response.set_cookie(
        key=COOKIE_ACCESS_NAME,
        value=access_token,
        max_age=access_max_age,
        httponly=True,
        secure=COOKIE_SECURE,
        samesite=COOKIE_SAMESITE,
        domain=COOKIE_DOMAIN,
        path="/",
    )

    # 2. HttpOnly Refresh Token (restricted to /api/v1/auth)
    response.set_cookie(
        key=COOKIE_REFRESH_NAME,
        value=refresh_token,
        max_age=refresh_max_age,
        httponly=True,
        secure=COOKIE_SECURE,
        samesite=COOKIE_SAMESITE,
        domain=COOKIE_DOMAIN,
        path="/api/v1/auth",
    )

    # 3. Optional CSRF cookie (readable by frontend for double-submit header verification)
    if csrf_token:
        response.set_cookie(
            key=COOKIE_CSRF_NAME,
            value=csrf_token,
            max_age=refresh_max_age,
            httponly=False,
            secure=COOKIE_SECURE,
            samesite=COOKIE_SAMESITE,
            domain=COOKIE_DOMAIN,
            path="/",
        )


def clear_auth_cookies(response: Response):
    """Clear all authentication cookies."""
    response.delete_cookie(
        key=COOKIE_ACCESS_NAME,
        domain=COOKIE_DOMAIN,
        path="/",
        httponly=True,
        secure=COOKIE_SECURE,
        samesite=COOKIE_SAMESITE,
    )
    response.delete_cookie(
        key=COOKIE_REFRESH_NAME,
        domain=COOKIE_DOMAIN,
        path="/api/v1/auth",
        httponly=True,
        secure=COOKIE_SECURE,
        samesite=COOKIE_SAMESITE,
    )
    response.delete_cookie(
        key=COOKIE_CSRF_NAME,
        domain=COOKIE_DOMAIN,
        path="/",
        httponly=False,
        secure=COOKIE_SECURE,
        samesite=COOKIE_SAMESITE,
    )
