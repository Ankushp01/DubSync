"""Database repositories package."""
from database.repositories.users import UsersRepository
from database.repositories.sessions import SessionsRepository
from database.repositories.projects import ProjectsRepository
from database.repositories.videos import VideosRepository
from database.repositories.transcripts import TranscriptsRepository
from database.repositories.translations import TranslationsRepository
from database.repositories.audio import GeneratedAudioRepository
from database.repositories.dubbed_videos import DubbedVideosRepository
from database.repositories.processing_logs import ProcessingLogsRepository
from database.repositories.jobs import ProcessingJobsRepository
from database.repositories.segments import SegmentsRepository
from database.repositories.evaluations import EvaluationsRepository

# Aliases
DubbingProjectsRepository = ProjectsRepository

__all__ = [
    "UsersRepository",
    "SessionsRepository",
    "ProjectsRepository",
    "DubbingProjectsRepository",
    "VideosRepository",
    "TranscriptsRepository",
    "TranslationsRepository",
    "GeneratedAudioRepository",
    "DubbedVideosRepository",
    "ProcessingLogsRepository",
    "ProcessingJobsRepository",
    "SegmentsRepository",
    "EvaluationsRepository",
]
