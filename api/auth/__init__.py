"""Authentication package."""
from api.auth.routes import router as auth_router
from api.auth.dependencies import get_current_user, get_optional_current_user, get_auth_service
from api.auth.service import AuthService
from api.auth.security import hash_password, verify_password, create_access_token, decode_access_token

__all__ = [
    "auth_router",
    "get_current_user",
    "get_optional_current_user",
    "get_auth_service",
    "AuthService",
    "hash_password",
    "verify_password",
    "create_access_token",
    "decode_access_token",
]
