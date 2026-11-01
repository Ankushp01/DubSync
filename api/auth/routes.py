"""Authentication API endpoints."""
from __future__ import annotations

from typing import Optional
from fastapi import APIRouter, Cookie, Depends, Header, HTTPException, Request, Response, status

from api.auth.dependencies import get_auth_service, get_current_user
from api.auth.schemas import (
    AuthResponse,
    ForgotPasswordRequest,
    ForgotPasswordResponse,
    LoginRequest,
    MessageResponse,
    RegisterRequest,
    ResendVerificationRequest,
    ResetPasswordRequest,
    UserResponse,
    VerifyEmailRequest,
)
from api.auth.security import COOKIE_REFRESH_NAME, clear_auth_cookies, set_auth_cookies
from api.auth.service import AuthService
from database.models import UserModel

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])


def _get_client_ip(request: Request) -> Optional[str]:
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else None


@router.post("/register", response_model=AuthResponse, status_code=status.HTTP_201_CREATED)
async def register(
    req: RegisterRequest,
    request: Request,
    response: Response,
    service: AuthService = Depends(get_auth_service),
):
    """Register a new user account with Argon2id password hashing."""
    ip = _get_client_ip(request)
    ua = request.headers.get("User-Agent")
    user, access_token, refresh_token, csrf_token, verify_token, delivery = await service.register(req, user_agent=ua, ip_address=ip)
    set_auth_cookies(response, access_token=access_token, refresh_token=refresh_token, csrf_token=csrf_token)
    return AuthResponse(
        user=user,
        access_token=access_token,
        token_type="bearer",
        expires_in=900,
        preview_verification_token=verify_token,
        email_sent=delivery.success if delivery else False,
        delivery_status="sent" if (delivery and delivery.success) else ((delivery.error_code or "delivery_failed") if delivery else "not_configured"),
    )


@router.post("/login", response_model=AuthResponse)
async def login(
    req: LoginRequest,
    request: Request,
    response: Response,
    service: AuthService = Depends(get_auth_service),
):
    """Authenticate with normalized email and password. Issues short-lived access token and refresh cookie."""
    ip = _get_client_ip(request)
    ua = request.headers.get("User-Agent")
    user, access_token, refresh_token, csrf_token = await service.login(req, user_agent=ua, ip_address=ip)
    set_auth_cookies(response, access_token=access_token, refresh_token=refresh_token, csrf_token=csrf_token)
    return AuthResponse(
        user=user,
        access_token=access_token,
        token_type="bearer",
        expires_in=900,
    )


@router.post("/refresh", response_model=AuthResponse)
async def refresh_session(
    request: Request,
    response: Response,
    refresh_token_cookie: Optional[str] = Cookie(None, alias=COOKIE_REFRESH_NAME),
    service: AuthService = Depends(get_auth_service),
):
    """Rotate refresh token and issue a fresh access token."""
    raw_token = request.headers.get("X-Refresh-Token") or refresh_token_cookie
    if not raw_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token missing from cookie or header.",
        )
    ip = _get_client_ip(request)
    ua = request.headers.get("User-Agent")
    user, new_access, new_refresh, new_csrf = await service.refresh_tokens(raw_token, user_agent=ua, ip_address=ip)
    set_auth_cookies(response, access_token=new_access, refresh_token=new_refresh, csrf_token=new_csrf)
    return AuthResponse(
        user=user,
        access_token=new_access,
        token_type="bearer",
        expires_in=900,
    )


@router.post("/logout", response_model=MessageResponse)
async def logout(
    response: Response,
    refresh_token_cookie: Optional[str] = Cookie(None, alias=COOKIE_REFRESH_NAME),
    service: AuthService = Depends(get_auth_service),
):
    """Revoke current refresh session and clear authentication cookies."""
    await service.logout(refresh_token_cookie)
    clear_auth_cookies(response)
    return MessageResponse(message="Successfully logged out.")


@router.post("/logout-all", response_model=MessageResponse)
async def logout_all(
    response: Response,
    current_user: UserModel = Depends(get_current_user),
    service: AuthService = Depends(get_auth_service),
):
    """Revoke all active sessions for the authenticated user and clear cookies."""
    await service.logout_all(current_user.user_id)
    clear_auth_cookies(response)
    return MessageResponse(message="All active sessions have been revoked.", status="all_sessions_revoked")


@router.get("/me", response_model=UserResponse)
async def get_me(current_user: UserModel = Depends(get_current_user)):
    """Return the profile of the currently authenticated user."""
    return UserResponse(
        user_id=current_user.user_id,
        name=current_user.name,
        display_name=getattr(current_user, "display_name", None) or current_user.name,
        email=current_user.email,
        role=getattr(current_user, "role", "user"),
        email_verified=current_user.email_verified,
        is_active=current_user.is_active,
        created_at=str(current_user.created_at),
        last_login_at=str(current_user.last_login_at) if current_user.last_login_at else None,
        preferences=current_user.preferences.model_dump() if hasattr(current_user.preferences, "model_dump") else dict(current_user.preferences),
    )


@router.post("/verify-email", response_model=MessageResponse)
async def verify_email(
    req: VerifyEmailRequest,
    service: AuthService = Depends(get_auth_service),
):
    """Verify email address using verification token."""
    await service.verify_email(req.token)
    return MessageResponse(message="Email address successfully verified.")


@router.post("/resend-verification", response_model=MessageResponse)
async def resend_verification(
    req: ResendVerificationRequest,
    request: Request,
    service: AuthService = Depends(get_auth_service),
):
    """Resend email verification token with rate limiting."""
    ip = _get_client_ip(request)
    delivery = await service.resend_verification(req.email, ip_address=ip)
    if delivery:
        if delivery.success:
            return MessageResponse(
                message="A new verification email has been sent to your address.",
                status="sent",
                delivery_status="sent",
            )
        else:
            return MessageResponse(
                message=f"Verification email could not be sent: {delivery.error_message}",
                status="delivery_failed",
                delivery_status=delivery.error_code or "failed",
                error_details=delivery.error_message,
            )
    return MessageResponse(
        message="If an unverified account exists for this email, a new verification link has been sent.",
        status="sent",
        delivery_status="sent",
    )


@router.post("/forgot-password", response_model=ForgotPasswordResponse)
async def forgot_password(
    req: ForgotPasswordRequest,
    request: Request,
    service: AuthService = Depends(get_auth_service),
):
    """Initiate password recovery without account enumeration."""
    ip = _get_client_ip(request)
    raw_token = await service.forgot_password(req, ip_address=ip)
    return ForgotPasswordResponse(
        message="If an account with that email exists, password reset instructions have been sent.",
        preview_token=raw_token,
    )


@router.post("/reset-password", response_model=MessageResponse)
async def reset_password(
    req: ResetPasswordRequest,
    service: AuthService = Depends(get_auth_service),
):
    """Reset password using single-use reset token and terminate all sessions."""
    await service.reset_password(req)
    return MessageResponse(message="Password successfully reset. Please log in with your new password.")
