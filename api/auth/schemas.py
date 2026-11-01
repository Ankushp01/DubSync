"""Pydantic schemas for authentication and user management."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional
from pydantic import BaseModel, EmailStr, Field, model_validator


class RegisterRequest(BaseModel):
    name: Optional[str] = None
    display_name: Optional[str] = None
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=128)

    @model_validator(mode="after")
    def check_name_present(self) -> RegisterRequest:
        if not self.name and not self.display_name:
            raise ValueError("Either name or display_name must be provided")
        if not self.name:
            self.name = self.display_name
        if not self.display_name:
            self.display_name = self.name
        return self


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class UserPreferencesSchema(BaseModel):
    default_source_language: str = "en"
    default_target_language: str = "hi"
    default_voice: Optional[str] = None


class UserResponse(BaseModel):
    user_id: str
    name: str
    display_name: str = ""
    email: str
    role: str = "user"
    email_verified: bool
    is_active: bool
    created_at: str
    last_login_at: Optional[str] = None
    preferences: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def ensure_display_name(self) -> UserResponse:
        if not self.display_name:
            self.display_name = self.name
        return self


class AuthResponse(BaseModel):
    user: UserResponse
    access_token: str
    token_type: str = "bearer"
    expires_in: int = 900
    preview_verification_token: Optional[str] = None


class UpdateUserRequest(BaseModel):
    name: Optional[str] = Field(None, min_length=2, max_length=100)
    display_name: Optional[str] = Field(None, min_length=2, max_length=100)
    preferences: Optional[Dict[str, Any]] = None


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str = Field(..., min_length=8, max_length=128)


class DeleteAccountRequest(BaseModel):
    password: str


class ForgotPasswordRequest(BaseModel):
    email: EmailStr


class ForgotPasswordResponse(BaseModel):
    message: str
    preview_token: Optional[str] = None
    success: bool = True


class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str = Field(..., min_length=8, max_length=128)


class VerifyEmailRequest(BaseModel):
    token: str


class ResendVerificationRequest(BaseModel):
    email: EmailStr


class MessageResponse(BaseModel):
    message: str
    status: Optional[str] = None
    success: bool = True
