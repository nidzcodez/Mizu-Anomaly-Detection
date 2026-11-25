"""Usage: python3 make_user.py nida   (prints a block to paste into .streamlit/secrets.toml)"""
import getpass, re, sys
import bcrypt

name = (sys.argv[1] if len(sys.argv) > 1 else input("Username: ")).strip().lower()
if not re.fullmatch(r"[a-z0-9_-]{2,32}", name):
    sys.exit("Username: 2-32 characters, letters, numbers, - or _ only.")
pw = getpass.getpass("Password (min 12 characters): ")
if len(pw) < 12:
    sys.exit("Too short. Use at least 12 characters.")
if pw != getpass.getpass("Repeat password: "):
    sys.exit("Passwords do not match.")
h = bcrypt.hashpw(pw.encode("utf-8")[:72], bcrypt.gensalt(rounds=12)).decode()
print(f'\n[auth.users.{name}]\npassword_hash = "{h}"\n')
