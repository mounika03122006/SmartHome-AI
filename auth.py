"""Single-admin authentication. Credentials are stored as a salted PBKDF2 hash in admin_credentials.json
(never in code). Create the admin once with:  python -m src.auth   (add --reset to replace it)."""
from __future__ import annotations
import getpass, hashlib, hmac, json, secrets, sys
from pathlib import Path

FILE = Path(__file__).resolve().parents[1] / "admin_credentials.json"
ITERATIONS = 200_000
MIN_PASSWORD = 8


def _hash(password: str, salt: bytes) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt, ITERATIONS)


def is_configured(path: Path = FILE) -> bool:
    return Path(path).exists()


def create_admin(username: str, password: str, path: Path = FILE, overwrite: bool = False) -> None:
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError("An admin already exists. Use --reset to replace it.")
    username = username.strip()
    if not username:
        raise ValueError("Username cannot be empty.")
    if len(password) < MIN_PASSWORD:
        raise ValueError(f"Password must be at least {MIN_PASSWORD} characters.")
    salt = secrets.token_bytes(16)
    path.write_text(json.dumps({"username": username, "salt": salt.hex(), "hash": _hash(password, salt).hex()}))


def verify(username: str, password: str, path: Path = FILE) -> bool:
    try:
        d = json.loads(Path(path).read_text())
        ok_user = hmac.compare_digest(username.strip().encode(), d["username"].encode())
        ok_pw = hmac.compare_digest(_hash(password, bytes.fromhex(d["salt"])), bytes.fromhex(d["hash"]))
        return ok_user and ok_pw
    except (FileNotFoundError, KeyError, ValueError):
        return False


def get_username(path: Path = FILE) -> str:
    try:
        return json.loads(Path(path).read_text())["username"]
    except (FileNotFoundError, KeyError, ValueError):
        return "Administrator"


def change_password(old: str, new: str, path: Path = FILE) -> None:
    d = json.loads(Path(path).read_text())
    if not verify(d["username"], old, path):
        raise PermissionError("Current password is incorrect.")
    create_admin(d["username"], new, path, overwrite=True)


if __name__ == "__main__":
    reset = "--reset" in sys.argv
    if is_configured() and not reset:
        sys.exit("Admin already exists. Run `python -m src.auth --reset` to replace it.")
    user = input("Choose admin username: ")
    pw = getpass.getpass(f"Choose password (min {MIN_PASSWORD} chars): ")
    if pw != getpass.getpass("Repeat password: "):
        sys.exit("Passwords do not match.")
    try:
        create_admin(user, pw, overwrite=reset)
    except (ValueError, FileExistsError) as e:
        sys.exit(str(e))
    print("Admin created. Keep admin_credentials.json private.")
