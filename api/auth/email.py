"""Email delivery service abstraction supporting Resend API with safe error handling and templating."""
from __future__ import annotations

import asyncio
import logging
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("dubsync.email")


@dataclass
class DeliveryResult:
    """Structured result returned by email delivery providers."""
    success: bool
    message_id: Optional[str] = None
    error_message: Optional[str] = None
    error_code: Optional[str] = None
    provider: str = "resend"

    def to_dict(self):
        return {
            "success": self.success,
            "message_id": self.message_id,
            "error_message": self.error_message,
            "error_code": self.error_code,
            "provider": self.provider,
        }


def mask_email(email: str) -> str:
    """Mask email for safe logging (e.g. j***e@example.com)."""
    if not email or "@" not in email:
        return "***"
    local, domain = email.split("@", 1)
    if len(local) <= 2:
        masked_local = local[0] + "*"
    else:
        masked_local = local[0] + "*" * (len(local) - 2) + local[-1]
    return f"{masked_local}@{domain}"


class BaseEmailProvider(ABC):
    """Abstract email delivery provider interface."""

    @abstractmethod
    async def send_email(
        self,
        to_email: str,
        subject: str,
        html_content: str,
        text_content: str,
    ) -> DeliveryResult:
        """Send an email and return delivery confirmation."""
        pass

    @property
    @abstractmethod
    def is_configured(self) -> bool:
        """Return True if provider has valid credentials configured."""
        pass


class ResendEmailProvider(BaseEmailProvider):
    """Transactional email provider powered by the official Resend API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        email_from: Optional[str] = None,
        frontend_url: Optional[str] = None,
    ):
        self.api_key = api_key or os.getenv("RESEND_API_KEY")
        self.email_from = email_from or os.getenv("EMAIL_FROM", "onboarding@resend.dev")
        self.frontend_url = (frontend_url or os.getenv("FRONTEND_URL", "http://localhost:3000")).rstrip("/")

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key and self.api_key.startswith("re_"))

    async def send_email(
        self,
        to_email: str,
        subject: str,
        html_content: str,
        text_content: str,
    ) -> DeliveryResult:
        masked = mask_email(to_email)
        if not self.is_configured:
            logger.warning(
                f"[ResendEmailProvider] Cannot deliver email to {masked}: "
                f"RESEND_API_KEY is not configured in backend environment."
            )
            return DeliveryResult(
                success=False,
                error_message="Email delivery service is not configured (missing RESEND_API_KEY).",
                error_code="PROVIDER_NOT_CONFIGURED",
                provider="resend",
            )

        def _sync_send() -> dict:
            import resend
            resend.api_key = self.api_key
            params = {
                "from": self.email_from,
                "to": to_email,
                "subject": subject,
                "html": html_content,
                "text": text_content,
            }
            return resend.Emails.send(params)

        try:
            # Execute synchronous SDK call in thread pool to preserve async loop
            response = await asyncio.to_thread(_sync_send)

            # Resend returns dict with 'id' on success
            if isinstance(response, dict) and "id" in response:
                msg_id = str(response["id"])
                logger.info(f"[ResendEmailProvider] Email delivered to {masked} (Resend ID: {msg_id})")
                return DeliveryResult(
                    success=True,
                    message_id=msg_id,
                    provider="resend",
                )
            else:
                logger.warning(f"[ResendEmailProvider] Unexpected response structure from Resend: {response}")
                return DeliveryResult(
                    success=False,
                    error_message=f"Unexpected response from Resend provider: {response}",
                    error_code="UNEXPECTED_RESPONSE",
                    provider="resend",
                )

        except Exception as exc:
            err_str = str(exc)
            logger.error(f"[ResendEmailProvider] Failed delivering email to {masked}: {err_str}")

            # Specific handling for Resend test account restrictions
            if "You can only send testing emails to your own email address" in err_str:
                clean_err = (
                    "Resend testing mode restriction: with 'onboarding@resend.dev', emails can only "
                    "be delivered to your verified Resend account email. Please verify a domain or use your account email."
                )
                return DeliveryResult(
                    success=False,
                    error_message=clean_err,
                    error_code="TESTING_RECIPIENT_RESTRICTION",
                    provider="resend",
                )
            elif "API key is invalid" in err_str or "unauthorized" in err_str.lower():
                return DeliveryResult(
                    success=False,
                    error_message="Resend API key is invalid or lacks necessary permissions.",
                    error_code="AUTHENTICATION_FAILED",
                    provider="resend",
                )
            elif "domain" in err_str.lower() and "verify" in err_str.lower():
                return DeliveryResult(
                    success=False,
                    error_message=f"Sender domain restriction: {err_str}",
                    error_code="SENDER_DOMAIN_UNVERIFIED",
                    provider="resend",
                )
            else:
                return DeliveryResult(
                    success=False,
                    error_message=f"Email delivery error: {err_str}",
                    error_code="PROVIDER_ERROR",
                    provider="resend",
                )


class EmailService:
    """High-level email service handling message templates and provider orchestration."""

    def __init__(self, provider: Optional[BaseEmailProvider] = None):
        self.provider = provider or ResendEmailProvider()
        self.frontend_url = (os.getenv("FRONTEND_URL", "http://localhost:3000")).rstrip("/")

    @property
    def is_configured(self) -> bool:
        return self.provider.is_configured

    async def send_verification_email(self, email: str, token: str) -> DeliveryResult:
        """Send account email verification link with branded HTML and plain-text fallback."""
        verify_url = f"{self.frontend_url}/verify-email?token={token}"
        subject = "Verify your DubSync account"

        html_content = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{subject}</title>
</head>
<body style="margin: 0; padding: 0; background-color: #090d16; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; color: #e2e8f0;">
  <table width="100%" cellpadding="0" cellspacing="0" style="background-color: #090d16; padding: 40px 20px;">
    <tr>
      <td align="center">
        <table width="100%" cellpadding="0" cellspacing="0" style="max-width: 540px; background-color: #111827; border: 1px solid #1f2937; border-radius: 16px; overflow: hidden; box-shadow: 0 10px 25px rgba(0, 0, 0, 0.5);">
          <!-- Header -->
          <tr>
            <td style="padding: 32px 32px 24px 32px; background: linear-gradient(135deg, rgba(99, 102, 241, 0.15) 0%, rgba(168, 85, 247, 0.15) 100%); border-bottom: 1px solid #1f2937; text-align: center;">
              <div style="display: inline-block; width: 44px; height: 44px; background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%); border-radius: 12px; line-height: 44px; text-align: center; color: #ffffff; font-size: 20px; font-weight: bold; margin-bottom: 12px;">DS</div>
              <h1 style="margin: 0; font-size: 22px; font-weight: 700; color: #f8fafc; letter-spacing: -0.025em;">Verify Your Email Address</h1>
              <p style="margin: 6px 0 0 0; font-size: 13px; color: #94a3b8;">Neural Video Dubbing & Audio Alignment</p>
            </td>
          </tr>
          <!-- Body -->
          <tr>
            <td style="padding: 32px;">
              <p style="margin: 0 0 16px 0; font-size: 15px; line-height: 1.6; color: #cbd5e1;">
                Welcome to <strong>DubSync</strong>! Please confirm your email address to complete your account setup and activate your AI video dubbing studio workspace.
              </p>
              <!-- Action Button -->
              <div style="text-align: center; margin: 32px 0;">
                <a href="{verify_url}" style="display: inline-block; background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%); color: #ffffff; text-decoration: none; font-size: 14px; font-weight: 600; padding: 14px 32px; border-radius: 10px; box-shadow: 0 4px 14px rgba(99, 102, 241, 0.4);">
                  Verify Email Address
                </a>
              </div>
              <p style="margin: 24px 0 8px 0; font-size: 13px; color: #94a3b8;">
                Button not working? You can also paste this secure link directly into your browser:
              </p>
              <div style="padding: 12px; background-color: #0b0f19; border: 1px solid #1e293b; border-radius: 8px; word-break: break-all; font-family: monospace; font-size: 11px; color: #818cf8;">
                {verify_url}
              </div>
              <p style="margin: 24px 0 0 0; font-size: 12px; color: #64748b; line-height: 1.5;">
                This single-use link expires in <strong>24 hours</strong>. If you did not create a DubSync account, you can safely ignore this message.
              </p>
            </td>
          </tr>
          <!-- Footer -->
          <tr>
            <td style="padding: 20px 32px; background-color: #0d121f; border-top: 1px solid #1f2937; text-align: center; font-size: 11px; color: #64748b;">
              &copy; DubSync AI &bull; Automated Neural Video Pipeline
            </td>
          </tr>
        </table>
      </td>
    </tr>
  </table>
</body>
</html>
"""

        text_content = (
            f"Verify your DubSync account\n\n"
            f"Please verify your email address by visiting this link:\n"
            f"{verify_url}\n\n"
            f"This link expires in 24 hours and can only be used once.\n"
            f"If you did not request this, you can ignore this email."
        )

        return await self.provider.send_email(email, subject, html_content, text_content)

    async def send_password_reset_email(self, email: str, token: str) -> DeliveryResult:
        """Send password reset link with branded HTML and plain-text fallback."""
        reset_url = f"{self.frontend_url}/reset-password?token={token}"
        subject = "Reset your DubSync password"

        html_content = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{subject}</title>
</head>
<body style="margin: 0; padding: 0; background-color: #090d16; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; color: #e2e8f0;">
  <table width="100%" cellpadding="0" cellspacing="0" style="background-color: #090d16; padding: 40px 20px;">
    <tr>
      <td align="center">
        <table width="100%" cellpadding="0" cellspacing="0" style="max-width: 540px; background-color: #111827; border: 1px solid #1f2937; border-radius: 16px; overflow: hidden; box-shadow: 0 10px 25px rgba(0, 0, 0, 0.5);">
          <!-- Header -->
          <tr>
            <td style="padding: 32px 32px 24px 32px; background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(249, 115, 22, 0.15) 100%); border-bottom: 1px solid #1f2937; text-align: center;">
              <div style="display: inline-block; width: 44px; height: 44px; background: linear-gradient(135deg, #ef4444 0%, #f97316 100%); border-radius: 12px; line-height: 44px; text-align: center; color: #ffffff; font-size: 20px; font-weight: bold; margin-bottom: 12px;">DS</div>
              <h1 style="margin: 0; font-size: 22px; font-weight: 700; color: #f8fafc; letter-spacing: -0.025em;">Password Reset Request</h1>
              <p style="margin: 6px 0 0 0; font-size: 13px; color: #94a3b8;">DubSync Account Security</p>
            </td>
          </tr>
          <!-- Body -->
          <tr>
            <td style="padding: 32px;">
              <p style="margin: 0 0 16px 0; font-size: 15px; line-height: 1.6; color: #cbd5e1;">
                We received a request to reset the password for your DubSync account. Click the button below to choose a new password:
              </p>
              <!-- Action Button -->
              <div style="text-align: center; margin: 32px 0;">
                <a href="{reset_url}" style="display: inline-block; background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%); color: #ffffff; text-decoration: none; font-size: 14px; font-weight: 600; padding: 14px 32px; border-radius: 10px; box-shadow: 0 4px 14px rgba(239, 68, 68, 0.4);">
                  Reset Password
                </a>
              </div>
              <p style="margin: 24px 0 8px 0; font-size: 13px; color: #94a3b8;">
                Alternatively, paste this URL into your browser:
              </p>
              <div style="padding: 12px; background-color: #0b0f19; border: 1px solid #1e293b; border-radius: 8px; word-break: break-all; font-family: monospace; font-size: 11px; color: #f87171;">
                {reset_url}
              </div>
              <p style="margin: 24px 0 0 0; font-size: 12px; color: #64748b; line-height: 1.5;">
                This link expires in <strong>1 hour</strong> and can only be used once. If you did not request a password reset, no changes have been made and you can ignore this email.
              </p>
            </td>
          </tr>
          <!-- Footer -->
          <tr>
            <td style="padding: 20px 32px; background-color: #0d121f; border-top: 1px solid #1f2937; text-align: center; font-size: 11px; color: #64748b;">
              &copy; DubSync AI &bull; Automated Neural Video Pipeline
            </td>
          </tr>
        </table>
      </td>
    </tr>
  </table>
</body>
</html>
"""

        text_content = (
            f"Reset your DubSync password\n\n"
            f"A password reset was requested for your DubSync account. Visit the following link to set a new password:\n"
            f"{reset_url}\n\n"
            f"This link expires in 1 hour and can only be used once.\n"
            f"If you did not request this reset, you can safely ignore this email."
        )

        return await self.provider.send_email(email, subject, html_content, text_content)


# Default singleton instance
email_service = EmailService()
