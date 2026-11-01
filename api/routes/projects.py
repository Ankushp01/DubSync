"""API routes for Project management in DubSync."""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from api.auth.dependencies import get_current_user, require_verified_user
from database.connection import db_manager, DatabaseUnavailableError
from database.models import UserModel
from database.repositories.projects import ProjectsRepository

router = APIRouter(prefix="/api/v1/projects", tags=["projects"])


class CreateProjectRequest(BaseModel):
    project_name: Optional[str] = None
    name: Optional[str] = None
    source_language: str = "en"
    target_languages: List[str] = Field(default_factory=lambda: ["hi"])
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def get_name(self) -> str:
        return self.project_name or self.name or "Untitled Project"


class UpdateProjectRequest(BaseModel):
    project_name: Optional[str] = None
    name: Optional[str] = None
    source_language: Optional[str] = None
    target_languages: Optional[List[str]] = None
    status: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def get_name(self) -> Optional[str]:
        return self.project_name or self.name


class ProjectResponse(BaseModel):
    project_id: str
    user_id: Optional[str] = None
    project_name: str
    source_language: str
    target_languages: List[str]
    status: str
    created_at: str
    updated_at: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PaginatedProjectsResponse(BaseModel):
    items: List[ProjectResponse]
    page: int
    limit: int
    total: int
    total_pages: int


def _get_repo() -> ProjectsRepository:
    if not db_manager.is_connected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"MongoDB Atlas is currently unavailable. ({db_manager.last_error or 'Not connected'})",
        )
    return ProjectsRepository()


@router.post("", response_model=ProjectResponse, status_code=status.HTTP_201_CREATED)
async def create_project(req: CreateProjectRequest, current_user: UserModel = Depends(get_current_user)):
    """Create a new dubbing project."""
    repo = _get_repo()
    doc = await repo.create_project(
        project_name=req.get_name(),
        user_id=current_user.user_id,
        source_language=req.source_language,
        target_languages=req.target_languages,
        metadata=req.metadata,
    )
    return ProjectResponse(**doc)


@router.get("", response_model=PaginatedProjectsResponse)
async def list_projects(
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1, le=100),
    current_user: UserModel = Depends(get_current_user),
):
    """List projects for the authenticated user with pagination."""
    repo = _get_repo()
    res = await repo.list_projects(user_id=current_user.user_id, page=page, limit=limit)
    return PaginatedProjectsResponse(
        items=[ProjectResponse(**item) for item in res["items"]],
        page=res["page"],
        limit=res["limit"],
        total=res["total"],
        total_pages=res["total_pages"],
    )


@router.get("/{project_id}", response_model=ProjectResponse)
async def get_project(project_id: str, current_user: UserModel = Depends(get_current_user)):
    """Retrieve details of a specific project."""
    repo = _get_repo()
    doc = await repo.get_by_id_for_user(project_id, current_user.user_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found")
    return ProjectResponse(**doc)


@router.patch("/{project_id}", response_model=ProjectResponse)
async def update_project(project_id: str, req: UpdateProjectRequest, current_user: UserModel = Depends(get_current_user)):
    """Update metadata or status of a project."""
    repo = _get_repo()
    updates = req.model_dump(exclude_unset=True)
    if not updates:
        raise HTTPException(status_code=400, detail="No fields provided to update")
    doc = await repo.update_project_for_user(project_id, current_user.user_id, updates)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found")
    return ProjectResponse(**doc)


@router.delete("/{project_id}")
async def delete_project(project_id: str, current_user: UserModel = Depends(get_current_user)):
    """Delete project metadata record."""
    repo = _get_repo()
    deleted = await repo.delete_project_for_user(project_id, current_user.user_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found")
    return {"status": "deleted", "project_id": project_id}
