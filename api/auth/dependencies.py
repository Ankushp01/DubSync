"""FastAPI dependency injection for authentication and ownership checks."""
from __future__ import annotations

from typing import Optional
from fastapi import Cookie, Depends, Header, HTTPException, Query, Request, status
import jwt

from api.auth.security import COOKIE_ACCESS_NAME, decode_access_token
from api.auth.service import AuthService
from database.connection import DatabaseUnavailableError
from database.models import UserModel
from database.repositories.users import UsersRepository

_users_repo = UsersRepository()
_auth_service = AuthService(users_repo=_users_repo)


def get_auth_service() -> AuthService:
    return _auth_service


async def get_current_user(
    request: Request,
    authorization: Optional[str] = Header(None),
    access_token_cookie: Optional[str] = Cookie(None, alias=COOKIE_ACCESS_NAME),
    token_query: Optional[str] = Query(None, alias="token"),
) -> UserModel:
    """Extract and validate the current user from Bearer header, HttpOnly cookie, or token query param."""
    token: Optional[str] = None

    # 1. Check Authorization header
    if authorization:
        parts = authorization.split()
        if len(parts) == 2 and parts[0].lower() == "bearer":
            token = parts[1]

    # 2. Check HttpOnly cookie
    if not token and access_token_cookie:
        token = access_token_cookie

    # 3. Check token query parameter (useful for EventSource / HTML5 media src)
    if not token and token_query:
        token = token_query

    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Please log in.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # 3. Decode JWT
    try:
        payload = decode_access_token(token)
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Access token expired. Please refresh your session.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token subject.",
        )

    # 4. Fetch user from repository
    try:
        user_doc = await _users_repo.get_by_id(user_id)
        if not user_doc:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User account no longer exists.",
            )
        if not user_doc.get("is_active", True):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User account is deactivated.",
            )
        return UserModel(**user_doc)
    except DatabaseUnavailableError:
        # Fallback to cryptographically verified JWT payload when database connection is offline
        return UserModel(
            user_id=user_id,
            name=payload.get("name") or payload.get("email") or "User",
            email=payload.get("email", ""),
            role=payload.get("role", "user"),
            is_active=True,
            email_verified=True,
        )


async def get_optional_current_user(
    request: Request,
    authorization: Optional[str] = Header(None),
    access_token_cookie: Optional[str] = Cookie(None, alias=COOKIE_ACCESS_NAME),
    token_query: Optional[str] = Query(None, alias="token"),
) -> Optional[UserModel]:
    """Optional user dependency for public endpoints with personalized context."""
    try:
        return await get_current_user(
            request=request,
            authorization=authorization,
            access_token_cookie=access_token_cookie,
            token_query=token_query,
        )
    except HTTPException:
        return None
