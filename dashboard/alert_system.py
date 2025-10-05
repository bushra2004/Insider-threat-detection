# dashboard/alert_system.py

import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from datetime import datetime

class AlertSystem:
    """
    Small, robust alert system:
     - Uses SMTP to send emails (supports attachments)
     - Keeps an in-memory alert_history list for debugging/display
     - Reads SMTP server + port if needed, but defaults to Gmail SMTP
    """

    def __init__(self, sender_email=None, password=None, smtp_server='smtp.gmail.com', smtp_port=587):
        # Prefer explicit args, else use environment variables
        self.sender_email = sender_email or os.getenv("ALERT_SENDER_EMAIL")
        self.password = password or os.getenv("ALERT_SENDER_PASSWORD")
        self.smtp_server = smtp_server or os.getenv("ALERT_SMTP_SERVER", "smtp.gmail.com")
        self.smtp_port = int(smtp_port or os.getenv("ALERT_SMTP_PORT", 587))
        self.alert_history = []  # list of dicts with timestamp, subject, recipients, status

    def _make_message(self, subject, message, recipients):
        msg = MIMEMultipart()
        msg['From'] = self.sender_email
        msg['To'] = ", ".join(recipients if isinstance(recipients, (list,tuple)) else [recipients])
        msg['Subject'] = subject
        msg.attach(MIMEText(message, 'plain'))
        return msg

    def send_email_alert(self, subject, message, recipients, attachments=None):
        """
        Send an email to recipients.
        - recipients: single email string or list of emails
        - attachments: list of file paths to attach (optional)
        """
        if isinstance(recipients, str):
            recipients = [recipients]

        if not self.sender_email or not self.password:
            err = "SMTP credentials not configured. Set ALERT_SENDER_EMAIL and ALERT_SENDER_PASSWORD environment variables."
            print("❌", err)
            self.alert_history.append({
                "timestamp": datetime.now(),
                "subject": subject,
                "recipients": recipients,
                "status": f"failed: {err}"
            })
            return False

        try:
            msg = self._make_message(subject, message, recipients)

            # Attach files (if provided)
            if attachments:
                for path in attachments:
                    try:
                        with open(path, "rb") as f:
                            part = MIMEApplication(f.read(), Name=path.split("/")[-1])
                            part['Content-Disposition'] = f'attachment; filename="{path.split("/")[-1]}"'
                            msg.attach(part)
                    except Exception as e:
                        print(f"Warning: failed to attach {path}: {e}")

            # Send using SMTP
            with smtplib.SMTP(self.smtp_server, self.smtp_port, timeout=30) as server:
                server.ehlo()
                server.starttls()
                server.ehlo()
                server.login(self.sender_email, self.password)
                server.send_message(msg)

            self.alert_history.append({
                "timestamp": datetime.now(),
                "subject": subject,
                "recipients": recipients,
                "status": "success"
            })
            print(f"✅ Alert sent to {recipients}")
            return True

        except Exception as e:
            print("❌ Failed to send alert:", e)
            self.alert_history.append({
                "timestamp": datetime.now(),
                "subject": subject,
                "recipients": recipients,
                "status": f"failed: {e}"
            })
            return False

    # Convenience wrapper for DataFrame-based alerts
    def check_for_alerts(self, df, to_emails, threshold=-0.3):
        """
        Checks df for critical anomalies and emails them.
        Expects df with columns: ['user','is_anomaly','anomaly_score','date', ...]
        to_emails: list or single email recipient
        """
        if isinstance(to_emails, str):
            to_emails = [to_emails]

        critical = df[(df.get("is_anomaly") == 1) & (df.get("anomaly_score") < threshold)]
        if critical.empty:
            return []

        results = []
        for _, row in critical.iterrows():
            subj = f"🚨 Critical anomaly: {row.get('user', 'unknown')}"
            body = (f"CRITICAL ANOMALY DETECTED\n\n"
                    f"User: {row.get('user')}\n"
                    f"Risk Score: {row.get('anomaly_score')}\n"
                    f"Date: {row.get('date')}\n\n"
                    f"Details:\n"
                    f"- Failed Logins: {row.get('failed_logins', 0)}\n"
                    f"- File Access: {row.get('file_access', 0)}\n\n"
                    f"Recommended action: Investigate and review user's access.")
            ok = self.send_email_alert(subj, body, to_emails)
            results.append({"user": row.get("user"), "sent": ok})
        return results
