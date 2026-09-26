"""Create local, ignored credentials for Langfuse. Never overwrites .env."""

from pathlib import Path
import secrets

destination = Path(__file__).resolve().parents[1] / ".env"
if destination.exists():
    print(f"{destination} already exists; leaving it unchanged")
    raise SystemExit(0)


def token(n=32):
    return secrets.token_urlsafe(n)


values = {
    "POSTGRES_PASSWORD": token(),
    "CLICKHOUSE_PASSWORD": token(),
    "REDIS_AUTH": token(),
    "MINIO_ROOT_PASSWORD": token(),
    "NEXTAUTH_SECRET": token(),
    "SALT": token(),
    "ENCRYPTION_KEY": secrets.token_hex(32),
    "LANGFUSE_INIT_USER_PASSWORD": token(18),
    "LANGFUSE_PUBLIC_KEY": "pk-lf-" + secrets.token_hex(16),
    "LANGFUSE_SECRET_KEY": "sk-lf-" + secrets.token_hex(32),
}
destination.write_text("\n".join(f"{key}={value}" for key, value in values.items()) + "\n")
destination.chmod(0o600)
print(f"Created {destination}; login is workshop@example.com and password is LANGFUSE_INIT_USER_PASSWORD in that file")
