"""User profile management routes."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Response, status

from api.auth.dependencies import get_auth_service, get_current_user
from api.auth.schemas import (
    ChangePasswordRequest,
    DeleteAccountRequest,
    MessageResponse,
    UpdateUserRequest,
    UserResponse,
)
from api.auth.security import clear_auth_cookies, verify_password
from api.auth.service import AuthService
from database.models import UserModel
from database.repositories.users import UsersRepository

router = APIRouter(prefix="/api/v1/users", tags=["users"])


@router.patch("/me", response_model=UserResponse)
async def update_my_profile(
    req: UpdateUserRequest,
    current_user: UserModel = Depends(get_current_user),
    service: AuthService = Depends(get_auth_service),
):
    """Update profile details (name, preferences) of the authenticated user."""
    updates = {}
    if req.name is not None:
        updates["name"] = req.name.strip()
    if req.preferences is not None:
        updates["preferences"] = req.preferences

    if not updates:
        raise HTTPException(status_code=400, detail="No fields provided to update.")

    updated_doc = await service.users_repo.update_user(current_user.user_id, updates)
    if not updated_doc:
        raise HTTPException(status_code=404, detail="User not found.")

    return UserResponse(
        user_id=updated_doc["user_id"],
        name=updated_doc["name"],
        email=updated_doc["email"],
        email_verified=updated_doc.get("email_verified", False),
        is_active=updated_doc.get("is_active", True),
        created_at=str(updated_doc["created_at"]),
        preferences=updated_doc.get("preferences", {}),
    )


@router.post("/me/change-password", response_model=UserResponse)
async def change_my_password(
    req: ChangePasswordRequest,
    current_user: UserModel = Depends(get_current_user),
    service: AuthService = Depends(get_auth_service),
):
    """Change user password, verifying current password and revoking all existing sessions."""
    return await service.change_password(current_user.user_id, req)


@router.delete("/me", response_model=MessageResponse)
async def delete_my_account(
    req: DeleteAccountRequest,
    response: Response,
    current_user: UserModel = Depends(get_current_user),
    service: AuthService = Depends(get_auth_service),
):
    """
    Delete the authenticated user's account and revoke all sessions.
    Requires current password confirmation.
    """
    user_doc = await service.users_repo.get_by_id(current_user.user_id)
    if not user_doc or not verify_password(user_doc.get("password_hash", ""), req.password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid password confirmation.")

    # 1. Revoke all active sessions
    await service.sessions_repo.revoke_all_user_sessions(current_user.user_id)

    # 2. Delete user account record
    await service.users_repo.delete_user(current_user.user_id)

    # 3. Clear auth cookies
    clear_auth_cookies(response)

    return MessageResponse(message="Account successfully deleted.")
