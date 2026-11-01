"""Authentication service layer: registration, credential validation, session rotation, and recovery."""
from __future__ import annotations

import logging
import os
import smtplib
import time
from datetime import datetime, timedelta, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Dict, Optional, Tuple

from fastapi import HTTPException, status

from api.auth.schemas import (
    ChangePasswordRequest,
    ForgotPasswordRequest,
    LoginRequest,
    RegisterRequest,
    ResetPasswordRequest,
    UpdateUserRequest,
    UserResponse,
)
from api.auth.security import (
    REFRESH_TOKEN_EXPIRE_DAYS,
    create_access_token,
    generate_csrf_token,
    generate_random_token,
    hash_password,
    hash_token,
    validate_password_strength,
    verify_password,
)
from database.models import utc_now
from database.repositories.sessions import SessionsRepository
from database.repositories.users import UsersRepository

logger = logging.getLogger("dubsync.auth")


# =========================================================================
# Rate Limiting Helper (In-Memory Sliding Window)
# =========================================================================
class RateLimiter:
    def __init__(self):
        self._attempts: Dict[str, list[float]] = {}

    def check_rate_limit(self, key: str, max_attempts: int, window_seconds: int) -> bool:
        """Returns True if within rate limit, False if limit exceeded."""
        now = time.time()
        timestamps = self._attempts.get(key, [])
        # Filter timestamps within window
        valid_timestamps = [t for t in timestamps if now - t < window_seconds]
        if len(valid_timestamps) >= max_attempts:
            self._attempts[key] = valid_timestamps
            return False
        valid_timestamps.append(now)
        self._attempts[key] = valid_timestamps
        return True

    def clear(self, key: str):
        self._attempts.pop(key, None)


rate_limiter = RateLimiter()


# =========================================================================
# Email Service Abstraction
# =========================================================================
class EmailService:
    def __init__(self):
        self.smtp_host = os.getenv("SMTP_HOST")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.smtp_user = os.getenv("SMTP_USER")
        self.smtp_pass = os.getenv("SMTP_PASS")
        self.smtp_from = os.getenv("SMTP_FROM", "noreply@dubsync.ai")
        self.frontend_url = os.getenv("FRONTEND_URL", "http://localhost:3000")

    @property
    def is_configured(self) -> bool:
        return bool(self.smtp_host and self.smtp_user and self.smtp_pass)

    def send_verification_email(self, email: str, token: str):
        verify_url = f"{self.frontend_url}/verify-email?token={token}"
        if not self.is_configured:
            logger.info(
                f"[EmailService] Verification token generated for '{email}'. "
                f"SMTP not configured in environment. In dev/testing, token is stored on user record."
            )
            return

        subject = "Verify your DubSync account"
        text_content = f"Please verify your DubSync account by clicking: {verify_url}\nThis link expires in 24 hours."
        self._send_email(email, subject, text_content)

    def send_password_reset_email(self, email: str, token: str):
        reset_url = f"{self.frontend_url}/reset-password?token={token}"
        if not self.is_configured:
            logger.info(
                f"[EmailService] Password reset token generated for '{email}'. "
                f"SMTP not configured in environment. In dev/testing, token is stored on user record."
            )
            return

        subject = "Reset your DubSync password"
        text_content = f"You requested a password reset. Click this link to set a new password: {reset_url}\nThis link expires in 1 hour."
        self._send_email(email, subject, text_content)

    def _send_email(self, to_email: str, subject: str, content: str):
        try:
            msg = MIMEMultipart()
            msg["From"] = self.smtp_from
            msg["To"] = to_email
            msg["Subject"] = subject
            msg.attach(MIMEText(content, "plain"))

            with smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=10) as server:
                server.starttls()
                server.login(self.smtp_user, self.smtp_pass)
                server.send_message(msg)
        except Exception as e:
            logger.warning(f"[EmailService] Failed to deliver email to {to_email}: {e}")


email_service = EmailService()


# =========================================================================
# Authentication Service Implementation
# =========================================================================
class AuthService:
    def __init__(self, users_repo: Optional[UsersRepository] = None, sessions_repo: Optional[SessionsRepository] = None):
        self.users_repo = users_repo or UsersRepository()
        self.sessions_repo = sessions_repo or SessionsRepository()

    async def register(
        self,
        req: RegisterRequest,
        user_agent: Optional[str] = None,
        ip_address: Optional[str] = None,
    ) -> Tuple[UserResponse, str, str, str, str]:
        """Register a new user account, create initial session, and return tokens."""
        # 1. Rate limiting: 10 attempts per 10 minutes per IP
        if ip_address and not rate_limiter.check_rate_limit(f"reg_{ip_address}", max_attempts=10, window_seconds=600):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many registration attempts. Please try again later.",
            )

        # 2. Email format & normalization
        normalized_email = req.email.strip().lower()

        # 3. Password strength check
        valid_pwd, pwd_error = validate_password_strength(req.password)
        if not valid_pwd:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=pwd_error,
            )

        # 4. Check for existing account (HTTP 409 Conflict)
        existing = await self.users_repo.get_by_email(normalized_email)
        if existing:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="An account with this email address already exists.",
            )

        # 5. Hash password with Argon2id
        pwd_hash = hash_password(req.password)

        # 6. Generate verification token
        raw_verify_token = generate_random_token(32)
        verify_token_hash = hash_token(raw_verify_token)
        verify_expires = utc_now() + timedelta(hours=24)

        # 7. Create user in database
        name_val = req.name or req.display_name or "User"
        user_doc = await self.users_repo.create_user(
            name=name_val.strip(),
            email=normalized_email,
            password_hash=pwd_hash,
            email_verified=False,
            is_active=True,
        )
        await self.users_repo.set_verification_token(user_doc["user_id"], verify_token_hash, verify_expires)

        # 8. Send verification email
        email_service.send_verification_email(normalized_email, raw_verify_token)

        # 9. Create session and issue tokens immediately
        raw_refresh_token = generate_random_token(48)
        refresh_hash = hash_token(raw_refresh_token)
        expires_at = utc_now() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)

        await self.sessions_repo.create_session(
            user_id=user_doc["user_id"],
            refresh_token_hash=refresh_hash,
            expires_at=expires_at,
            user_agent=user_agent,
            ip_address=ip_address,
        )

        access_token = create_access_token(
            subject=user_doc["user_id"],
            claims={
                "email": user_doc["email"],
                "role": "user",
            },
        )
        csrf_token = generate_csrf_token()

        user_response = UserResponse(
            user_id=user_doc["user_id"],
            name=user_doc["name"],
            display_name=user_doc["name"],
            email=user_doc["email"],
            email_verified=False,
            is_active=True,
            created_at=str(user_doc["created_at"]),
            preferences=user_doc.get("preferences", {}),
        )
        return user_response, access_token, raw_refresh_token, csrf_token, raw_verify_token

    async def login(
        self,
        req: LoginRequest,
        user_agent: Optional[str] = None,
        ip_address: Optional[str] = None,
    ) -> Tuple[UserResponse, str, str, str]:
        """Authenticate user, create session, and return tokens."""
        normalized_email = req.email.strip().lower()

        # Rate limiting: 5 failed attempts per 5 minutes per email/IP
        rate_key = f"login_{ip_address or 'anon'}_{normalized_email}"
        if not rate_limiter.check_rate_limit(rate_key, max_attempts=5, window_seconds=300):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many failed login attempts. Please wait 5 minutes before trying again.",
            )

        # Retrieve user
        user = await self.users_repo.get_by_email(normalized_email)
        if not user or not user.get("password_hash"):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password.",
            )

        # Verify Argon2id hash
        if not verify_password(user["password_hash"], req.password):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password.",
            )

        # Reject disabled/inactive accounts
        if not user.get("is_active", True):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="This account has been disabled. Please contact support.",
            )

        # Clear rate limit on successful authentication
        rate_limiter.clear(rate_key)

        # Update last login
        await self.users_repo.update_last_login(user["user_id"])

        # Create JWT access token
        access_token = create_access_token(user["user_id"], user["email"])

        # Create rotating refresh token
        raw_refresh_token = generate_random_token(48)
        refresh_hash = hash_token(raw_refresh_token)
        refresh_expires = utc_now() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)

        await self.sessions_repo.create_session(
            user_id=user["user_id"],
            refresh_token_hash=refresh_hash,
            expires_at=refresh_expires,
            user_agent=user_agent,
            ip_address=ip_address,
        )

        csrf_token = generate_random_token(24)

        user_response = UserResponse(
            user_id=user["user_id"],
            name=user["name"],
            email=user["email"],
            email_verified=user.get("email_verified", False),
            is_active=user.get("is_active", True),
            created_at=str(user["created_at"]),
            preferences=user.get("preferences", {}),
        )

        return user_response, access_token, raw_refresh_token, csrf_token

    async def refresh_tokens(
        self,
        raw_refresh_token: str,
        user_agent: Optional[str] = None,
        ip_address: Optional[str] = None,
    ) -> Tuple[UserResponse, str, str, str]:
        """Rotate refresh token and issue new access token. Detects token reuse."""
        if not raw_refresh_token:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Refresh token missing.")

        token_hash = hash_token(raw_refresh_token)
        session = await self.sessions_repo.get_by_token_hash(token_hash)

        if not session:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired session.")

        # Token reuse detection: if a revoked refresh token is presented, revoke all active sessions for this user
        if session.get("revoked_at") is not None:
            user_id = session["user_id"]
            await self.sessions_repo.revoke_all_user_sessions(user_id)
            logger.warning(
                f"[SECURITY ALERT] Refresh token reuse detected for user '{user_id}'. "
                f"All active sessions have been terminated."
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Security alert: Token reuse detected. All active sessions have been terminated. Please log in again.",
            )

        # Check expiration
        expires_at = session.get("expires_at")
        if isinstance(expires_at, str):
            expires_at = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
        if expires_at and expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)

        if expires_at and utc_now() > expires_at:
            await self.sessions_repo.revoke_session(session["session_id"])
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Refresh session has expired.")

        # Check user status
        user = await self.users_repo.get_by_id(session["user_id"])
        if not user or not user.get("is_active", True):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User account is inactive.")

        # Revoke the used session (Rotation)
        await self.sessions_repo.revoke_session(session["session_id"])

        # Create replacement session
        new_raw_refresh = generate_random_token(48)
        new_refresh_hash = hash_token(new_raw_refresh)
        new_refresh_expires = utc_now() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)

        await self.sessions_repo.create_session(
            user_id=user["user_id"],
            refresh_token_hash=new_refresh_hash,
            expires_at=new_refresh_expires,
            user_agent=user_agent or session.get("user_agent"),
            ip_address=ip_address or session.get("ip_address"),
        )

        new_access_token = create_access_token(user["user_id"], user["email"])
        new_csrf = generate_random_token(24)

        user_response = UserResponse(
            user_id=user["user_id"],
            name=user["name"],
            email=user["email"],
            email_verified=user.get("email_verified", False),
            is_active=user.get("is_active", True),
            created_at=str(user["created_at"]),
            preferences=user.get("preferences", {}),
        )

        return user_response, new_access_token, new_raw_refresh, new_csrf

    async def logout(self, raw_refresh_token: Optional[str]):
        """Revoke the current refresh token session."""
        if raw_refresh_token:
            token_hash = hash_token(raw_refresh_token)
            session = await self.sessions_repo.get_by_token_hash(token_hash)
            if session:
                await self.sessions_repo.revoke_session(session["session_id"])

    async def logout_all(self, user_id: str):
        """Revoke all active sessions for the user."""
        await self.sessions_repo.revoke_all_user_sessions(user_id)

    async def change_password(self, user_id: str, req: ChangePasswordRequest) -> UserResponse:
        """Change user password, verifying current password and revoking all other sessions."""
        user = await self.users_repo.get_by_id(user_id)
        if not user:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")

        # Verify old password
        if not verify_password(user.get("password_hash", ""), req.current_password):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Current password is incorrect.")

        # Validate new password strength
        valid_pwd, pwd_error = validate_password_strength(req.new_password)
        if not valid_pwd:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=pwd_error)

        # Update password hash
        new_hash = hash_password(req.new_password)
        await self.users_repo.update_password(user_id, new_hash)

        # Revoke all existing sessions
        await self.sessions_repo.revoke_all_user_sessions(user_id)

        updated_user = await self.users_repo.get_by_id(user_id)
        return UserResponse(
            user_id=updated_user["user_id"],
            name=updated_user["name"],
            email=updated_user["email"],
            email_verified=updated_user.get("email_verified", False),
            is_active=updated_user.get("is_active", True),
            created_at=str(updated_user["created_at"]),
            preferences=updated_user.get("preferences", {}),
        )

    async def forgot_password(self, req: ForgotPasswordRequest, ip_address: Optional[str] = None) -> Optional[str]:
        """Initiate password reset without leaking account existence."""
        normalized_email = req.email.strip().lower()

        # Rate limit: 3 requests per 15 minutes per email
        rate_key = f"reset_{normalized_email}"
        if not rate_limiter.check_rate_limit(rate_key, max_attempts=3, window_seconds=900):
            return None

        user = await self.users_repo.get_by_email(normalized_email)
        if user and user.get("is_active", True):
            raw_token = generate_random_token(32)
            token_hash = hash_token(raw_token)
            expires = utc_now() + timedelta(hours=1)
            await self.users_repo.set_reset_token(user["user_id"], token_hash, expires)
            email_service.send_password_reset_email(normalized_email, raw_token)
            return raw_token
        return None

    async def reset_password(self, req: ResetPasswordRequest):
        """Reset password using single-use reset token and revoke all user sessions."""
        token_hash = hash_token(req.token)
        user = await self.users_repo.get_by_reset_token_hash(token_hash)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid or expired password reset link.",
            )

        valid_pwd, pwd_error = validate_password_strength(req.new_password)
        if not valid_pwd:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=pwd_error)

        new_hash = hash_password(req.new_password)
        await self.users_repo.update_password(user["user_id"], new_hash)
        # Revoke all active sessions
        await self.sessions_repo.revoke_all_user_sessions(user["user_id"])

    async def verify_email(self, token: str) -> bool:
        """Verify email address using verification token."""
        token_hash = hash_token(token)
        user = await self.users_repo.verify_email_token(token_hash)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid or expired email verification link.",
            )
        return True

    async def resend_verification(self, email: str):
        """Resend email verification token without leaking account existence."""
        normalized_email = email.strip().lower()
        user = await self.users_repo.get_by_email(normalized_email)
        if user and not user.get("email_verified", False):
            raw_token = generate_random_token(32)
            token_hash = hash_token(raw_token)
            expires = utc_now() + timedelta(hours=24)
            await self.users_repo.set_verification_token(user["user_id"], token_hash, expires)
            email_service.send_verification_email(normalized_email, raw_token)
