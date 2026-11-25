"""MIZU login gate: bcrypt passwords, rate limiting (user + IP + global), roles, audit log, session timeouts."""
import hashlib
import math
import sqlite3
import time
from contextlib import closing
from pathlib import Path

import bcrypt
import streamlit as st

import login_fx
import streamlit.components.v1 as components

DB_PATH = Path(__file__).parent / ".auth_state.db"

BLUE = "#356dc1"          # MIZU blue: change this one value to retune the login page

MAX_FAILS = 5
IP_MAX_FAILS = 20
WINDOW_SECONDS = 15 * 60
LOCK_BASE_SECONDS = 60
LOCK_MAX_SECONDS = 3600
GLOBAL_PER_MINUTE = 30
IDLE_TIMEOUT = 30 * 60
ABSOLUTE_TIMEOUT = 8 * 3600
MAX_FIELD_LEN = 128
MIN_PASSWORD_LEN = 10

ROLES = {"viewer": 1, "operator": 2, "manager": 3}

_LOGO_SVG = (
    '<svg width="52" height="52" viewBox="0 0 48 48" xmlns="http://www.w3.org/2000/svg">'
    '<defs><linearGradient id="mzg" x1="0" y1="0" x2="1" y2="1">'
    '<stop offset="0" stop-color="#4b8ae6"/><stop offset="1" stop-color="#2f62b3"/></linearGradient></defs>'
    '<rect width="48" height="48" rx="12" fill="url(#mzg)"/>'
    '<path d="M24 9c6.2 7.3 10.2 12.2 10.2 17.4a10.2 10.2 0 0 1-20.4 0C13.8 21.2 17.8 16.3 24 9z" fill="#fff"/>'
    '<path d="M19.6 27.4a4.6 4.6 0 0 0 8.8 0" fill="none" stroke="#356dc1" stroke-width="2.6" stroke-linecap="round"/>'
    '</svg>'
)

_DUMMY_HASH = bcrypt.hashpw(b"not-a-real-password", bcrypt.gensalt(rounds=12))

_USER_ICON = ("%3Csvg xmlns='http://www.w3.org/2000/svg' width='20' height='20' viewBox='0 0 24 24' fill='none' "
              "stroke='%23356dc1' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E"
              "%3Ccircle cx='12' cy='8' r='4'/%3E%3Cpath d='M4 21c0-4 4-7 8-7s8 3 8 7'/%3E%3C/svg%3E")
_LOCK_ICON = ("%3Csvg xmlns='http://www.w3.org/2000/svg' width='20' height='20' viewBox='0 0 24 24' fill='none' "
              "stroke='%23356dc1' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E"
              "%3Crect x='5' y='11' width='14' height='10' rx='2'/%3E%3Cpath d='M8 11V7a4 4 0 0 1 8 0v4'/%3E%3C/svg%3E")

_LOGIN_CSS = """
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] { background: #f8f9fc !important; background-image: none !important; }
[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"] { display: none !important; }
.stApp .block-container { max-width: 1000px !important; padding-top: 9vh !important; }

.stApp .st-key-mizu-card-info, .stApp .st-key-mizu-card-login {
    background: #ffffff !important;
    border: 1px solid #d9e1ee !important;
    border-top: 3px solid __BLUE__ !important;
    border-radius: 14px !important;
    box-shadow: 0 1px 3px rgba(31, 42, 68, .06) !important;
    padding: 1.8rem 2rem 1.6rem !important;
}

.mg-top { display: flex; align-items: center; gap: 1rem; }
.mg-logo { width: 48px; height: 48px; background: __BLUE__; border-radius: 12px; display: flex; align-items: center; justify-content: center; flex: 0 0 auto; }
.mg-brand { font-size: 2.1rem; font-weight: 800; color: __BLUE__; line-height: 1.05; }
.mg-tag { color: #5b6678; font-size: .95rem; }
.mg-h { font-size: .8rem; font-weight: 700; color: #3f4d66; letter-spacing: .14em; text-transform: uppercase;
        border-left: 3px solid __BLUE__; padding-left: .6rem; margin: 1.6rem 0 .8rem; }
.mg-point { display: flex; gap: .8rem; align-items: flex-start; color: #1f2a44; font-size: 1rem; margin: .65rem 0; }
.mg-num { flex: 0 0 auto; width: 1.6rem; height: 1.6rem; border-radius: 50%; background: __BLUE__; color: #fff;
          font-weight: 700; font-size: .85rem; display: flex; align-items: center; justify-content: center; }
.mg-note { color: #5b6678; font-size: .85rem; line-height: 1.5; margin-top: 1.1rem; }
.mg-signlogo { display: flex; justify-content: center; margin: 0 0 .7rem; }
.mg-sign { text-align: center; font-size: 1.6rem; font-weight: 800; color: #1f2a44; margin: 0 0 .2rem; }
.mg-sub { text-align: center; color: #5b6678; font-size: .9rem; margin: 0 0 1.1rem; }

.stApp .st-key-mizu-card-login [data-testid="stWidgetLabel"] p { color: #1f2a44 !important; font-weight: 600; font-size: .9rem; }
.stApp .st-key-mizu-card-login [data-testid="InputInstructions"] { display: none !important; }
.stApp .st-key-mizu-user [data-baseweb="input"], .stApp .st-key-mizu-pass [data-baseweb="input"] {
    background: #fff !important; border-radius: 10px !important; border: 1px solid #cfd9ea !important; min-height: 42px !important;
}
.stApp .st-key-mizu-user [data-baseweb="base-input"], .stApp .st-key-mizu-pass [data-baseweb="base-input"] { background: transparent !important; border-radius: 10px !important; }
.stApp .st-key-mizu-user [data-baseweb="input"]:focus-within, .stApp .st-key-mizu-pass [data-baseweb="input"]:focus-within {
    border-color: __BLUE__ !important; box-shadow: 0 0 0 3px rgba(53, 109, 193, .18) !important;
}
.stApp .st-key-mizu-user input, .stApp .st-key-mizu-pass input {
    color: #1f2a44 !important; font-size: .95rem !important; padding: .35rem .8rem .35rem 2.7rem !important;
    background-repeat: no-repeat !important; background-position: .9rem center !important; background-size: 18px !important;
}
.stApp .st-key-mizu-user input { background-image: url("data:image/svg+xml;utf8,__USER_ICON__") !important; }
.stApp .st-key-mizu-pass input { background-image: url("data:image/svg+xml;utf8,__LOCK_ICON__") !important; }
.stApp .st-key-mizu-pass button { background: transparent !important; color: #5b6678 !important; }

.stApp .st-key-mizu-card-login [data-testid="stForm"] { border: none !important; padding: 0 !important; background: transparent !important; }
.stApp .st-key-mizu-card-login [data-testid="stFormSubmitButton"] { width: 100% !important; }
.stApp .st-key-mizu-card-login [data-testid="stFormSubmitButton"] button {
    width: 100% !important; border-radius: 10px !important; background: __BLUE__ !important; color: #fff !important;
    font-weight: 700 !important; font-size: 1rem !important; border: none !important; padding: .55rem 1rem !important;
}
.stApp .st-key-mizu-card-login [data-testid="stFormSubmitButton"] button:hover { background: #2c5aa3 !important; }
.stApp .st-key-mizu-card-login [data-testid="stAlert"] { border-radius: 10px !important; }

@media (max-width: 640px) {
    .stApp .block-container { padding-top: 3vh !important; }
    .stApp .st-key-mizu-card-info, .stApp .st-key-mizu-card-login { padding: 1.4rem 1.2rem 1.2rem !important; }
}
""".replace("__USER_ICON__", _USER_ICON).replace("__LOCK_ICON__", _LOCK_ICON).replace("__BLUE__", BLUE)


# ---------- storage ----------
def _db():
    con = sqlite3.connect(DB_PATH, timeout=5)
    con.execute("CREATE TABLE IF NOT EXISTS attempts (key TEXT, ts REAL, ok INTEGER)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_key_ts ON attempts (key, ts)")
    con.execute("CREATE TABLE IF NOT EXISTS pw_override (username TEXT PRIMARY KEY, hash TEXT, base TEXT, ts REAL)")
    con.execute("CREATE TABLE IF NOT EXISTS events (ts REAL, user TEXT, action TEXT, detail TEXT)")
    return con


def _key(name: str, kind: str = "u") -> str:
    return kind + ":" + hashlib.sha256(name.encode("utf-8")).hexdigest()[:24]


def _record(name: str, ok: bool, kind: str = "u") -> None:
    with closing(_db()) as con:
        con.execute("INSERT INTO attempts VALUES (?, ?, ?)", (_key(name, kind), time.time(), int(ok)))
        con.commit()


def _lock_remaining(name: str, kind: str = "u", max_fails: int = MAX_FAILS) -> int:
    """Seconds to wait before another attempt (0 if allowed)."""
    now = time.time()
    k = _key(name, kind)
    with closing(_db()) as con:
        con.execute("DELETE FROM attempts WHERE ts < ?", (now - 86400,))
        con.commit()
        last_ok = con.execute("SELECT MAX(ts) FROM attempts WHERE key=? AND ok=1", (k,)).fetchone()[0] or 0
        since = max(now - WINDOW_SECONDS, last_ok)
        fails, last_fail = con.execute(
            "SELECT COUNT(*), MAX(ts) FROM attempts WHERE key=? AND ok=0 AND ts>?", (k, since)
        ).fetchone()
        recent_all = con.execute(
            "SELECT COUNT(*) FROM attempts WHERE key LIKE 'u:%' AND ts>?", (now - 60,)
        ).fetchone()[0]
    wait = 0.0
    if fails >= max_fails:
        lock = min(LOCK_BASE_SECONDS * 2 ** min(fails - max_fails, 10), LOCK_MAX_SECONDS)
        wait = max(wait, (last_fail or 0) + lock - now)
    if recent_all >= GLOBAL_PER_MINUTE:
        wait = max(wait, 30)
    return math.ceil(wait) if wait > 0 else 0


def _client_ip() -> str:
    """Real client IP when behind a proxy such as Cloudflare. Empty when running locally."""
    try:
        h = st.context.headers
        cf = h.get("CF-Connecting-IP", "")
        if cf:
            return cf.strip()
        xff = h.get("X-Forwarded-For", "")
        return xff.split(",")[-1].strip() if xff else ""
    except Exception:
        return ""


# ---------- audit log ----------
def audit(action: str, detail: str = "", user: str | None = None) -> None:
    """Record an event. Call from app.py too, e.g. audit('upload', file.name)."""
    who = user or (st.session_state.get("auth") or {}).get("user", "-")
    try:
        with closing(_db()) as con:
            con.execute("INSERT INTO events VALUES (?, ?, ?, ?)", (time.time(), who, action[:40], str(detail)[:200]))
            con.commit()
    except Exception:
        pass


def audit_rows(limit: int = 200) -> list:
    """Latest events as (time, user, action, detail). Managers only."""
    if not can("manager"):
        return []
    with closing(_db()) as con:
        rows = con.execute("SELECT ts, user, action, detail FROM events ORDER BY ts DESC LIMIT ?", (limit,)).fetchall()
    return [(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(r[0])), r[1], r[2], r[3]) for r in rows]


# ---------- users and passwords ----------
def _users() -> dict:
    try:
        return dict(st.secrets["auth"]["users"])
    except Exception:
        return {}


def _role_of(rec: dict) -> str:
    r = str((rec or {}).get("role", "viewer")).strip().lower()
    return r if r in ROLES else "viewer"


def _stored_hash(username: str):
    rec = _users().get(username)
    base = rec.get("password_hash") if rec else None
    if not base:
        return None
    with closing(_db()) as con:
        row = con.execute("SELECT hash, base FROM pw_override WHERE username=?", (username,)).fetchone()
    # An in-app password change only counts while secrets.toml still holds the hash it was based on,
    # so if you reset a password in secrets.toml, the reset wins.
    if row and row[1] == base:
        return row[0]
    return base


def _verify(username: str, password: str) -> bool:
    rec = _users().get(username)
    stored = _stored_hash(username) if rec else None
    hashed = stored.encode("utf-8") if stored else _DUMMY_HASH
    try:
        match = bcrypt.checkpw(password.encode("utf-8")[:72], hashed)
    except ValueError:
        match = False
    return bool(rec) and bool(stored) and match


# ---------- roles ----------
def can(min_role: str) -> bool:
    """True if the signed-in user's role is at least min_role ('viewer' < 'operator' < 'manager')."""
    a = st.session_state.get("auth") or {}
    return ROLES.get(a.get("role", "viewer"), 1) >= ROLES[min_role]


def require_role(min_role: str) -> None:
    """Stops the page with a message if the user's role is too low."""
    if not can(min_role):
        st.warning(f"Your account can't do this. It needs the {min_role} role. Ask a manager.")
        st.stop()


def current_user_info() -> dict:
    """Returns {'user', 'name', 'role'} for the signed-in user, or {}."""
    return dict(st.session_state.get("auth") or {})


# ---------- login page ----------
_INFO_HTML = """
<div class="mg-top">
  __LOGO__
  <div><div class="mg-brand">MIZU</div><div class="mg-tag">Water systems monitoring</div></div>
</div>
<div class="mg-h">What MIZU does</div>
<div class="mg-point"><span class="mg-num">1</span><span>Checks 38 water-treatment measurements together.</span></div>
<div class="mg-point"><span class="mg-num">2</span><span>Flags unusual readings using two models.</span></div>
<div class="mg-point"><span class="mg-num">3</span><span>Shows which sensors contributed to each flag.</span></div>
<div class="mg-note">A decision-support tool for wastewater treatment. It does not certify drinking-water safety or control plant equipment.</div>
"""

_NOTE_HTML = """
<div class="mg-note">Authorised staff only. Data you upload is treated as confidential: it is used only to produce
your results in this session and is not shared. Accounts are created by your administrator.</div>
"""

_EXPIRED_MSG = {
    "idle": "You were signed out after 30 minutes of inactivity. Please sign in again.",
    "limit": "Your 8-hour session ended. Please sign in again.",
}


def _render_login() -> None:
    st.markdown("<style>" + _LOGIN_CSS + login_fx.CSS + "</style>", unsafe_allow_html=True)
    left, right = st.columns([1.15, 1], gap="large")
    with left:
        with st.container(key="mizu-card-info"):
            st.markdown(_INFO_HTML.replace("__LOGO__", _LOGO_SVG), unsafe_allow_html=True)
    with right:
        with st.container(key="mizu-card-login"):
            st.markdown('<div class="mg-signlogo">' + _LOGO_SVG + '</div><div class="mg-sign">Sign in</div><div class="mg-sub">Use the account your administrator created for you.</div>',
                        unsafe_allow_html=True)
            reason = st.session_state.pop("_auth_expired", None)
            if reason:
                st.info(_EXPIRED_MSG.get(reason, _EXPIRED_MSG["limit"]))
            with st.form("mizu_login"):
                u = st.text_input("Username", max_chars=MAX_FIELD_LEN, key="mizu-user")
                p = st.text_input("Password", type="password", max_chars=MAX_FIELD_LEN, key="mizu-pass")
                try:
                    go = st.form_submit_button("Sign in", width="stretch")
                except TypeError:
                    go = st.form_submit_button("Sign in", use_container_width=True)
            if go:
                _handle_submit(u, p)
            st.markdown(_NOTE_HTML, unsafe_allow_html=True)
    login_fx.run()


def _handle_submit(u, p) -> None:
    username = (u or "").strip().lower()[:MAX_FIELD_LEN]
    if not username or not p:
        st.error("Enter your username and password.")
        return
    ip = _client_ip()
    # Rate-limit checks run BEFORE the slow password check, so floods can't burn CPU on bcrypt.
    wait = _lock_remaining(username)
    if ip:
        wait = max(wait, _lock_remaining(ip, "i", IP_MAX_FAILS))
    if wait > 0:
        st.error(f"Too many attempts. Try again in {max(1, math.ceil(wait / 60))} minute(s).")
        return
    if _verify(username, p):
        _record(username, True)
        rec = _users().get(username) or {}
        now = time.time()
        st.session_state["auth"] = {
            "user": username, "name": rec.get("name", username), "role": _role_of(rec),
            "login_ts": now, "last_seen": now,
        }
        audit("login", user=username)
        st.rerun()
    else:
        _record(username, False)
        if ip:
            _record(ip, False, "i")
        audit("login_failed", user=username if username in _users() else "unknown")
        time.sleep(1)
        st.error("Incorrect username or password.")


def require_login() -> str:
    """Call right after st.set_page_config. Stops the script until the user is signed in."""
    if not _users():
        st.error("Sign-in is not configured. Add users to .streamlit/secrets.toml.")
        st.stop()
    a = st.session_state.get("auth")
    now = time.time()
    if a:
        reason = None
        if now - a["login_ts"] > ABSOLUTE_TIMEOUT:
            reason = "limit"
        elif now - a["last_seen"] > IDLE_TIMEOUT:
            reason = "idle"
        if reason:
            audit("session_expired", reason, user=a["user"])
            st.session_state.clear()
            st.session_state["_auth_expired"] = reason
        else:
            a["last_seen"] = now
            return a["user"]
    _render_login()
    st.stop()


# ---------- in-app account panel ----------
def _password_form(a: dict) -> None:
    with st.form("mizu_pw", clear_on_submit=True):
        cur = st.text_input("Current password", type="password", max_chars=MAX_FIELD_LEN)
        new = st.text_input("New password", type="password", max_chars=MAX_FIELD_LEN)
        conf = st.text_input("Confirm new password", type="password", max_chars=MAX_FIELD_LEN)
        go = st.form_submit_button("Update password")
    if not go:
        return
    user = a["user"]
    wait = _lock_remaining(user)
    if wait > 0:
        st.error(f"Too many attempts. Try again in {max(1, math.ceil(wait / 60))} minute(s).")
        return
    if not cur or not new:
        st.error("Fill in all fields.")
        return
    if not _verify(user, cur):
        _record(user, False)
        st.error("Current password is incorrect.")
        return
    _record(user, True)
    if len(new) < MIN_PASSWORD_LEN:
        st.error(f"New password must be at least {MIN_PASSWORD_LEN} characters.")
    elif len(new.encode("utf-8")) > 72:
        st.error("New password is too long (72 bytes maximum).")
    elif new != conf:
        st.error("The new passwords don't match.")
    elif new == cur or new.lower() == user:
        st.error("Choose a password different from your current one and your username.")
    else:
        base = (_users().get(user) or {}).get("password_hash")
        h = bcrypt.hashpw(new.encode("utf-8")[:72], bcrypt.gensalt(rounds=12)).decode("utf-8")
        with closing(_db()) as con:
            con.execute("INSERT OR REPLACE INTO pw_override VALUES (?, ?, ?, ?)", (user, h, base, time.time()))
            con.commit()
        audit("password_changed")
        st.success("Password updated.")


def logout_button(label: str = "Sign out") -> None:
    """Shows 'Name · Role', a change-password section and the sign-out button."""
    a = current_user_info()
    if not a:
        return
    st.caption(f"{a.get('name', a['user'])} · {a.get('role', 'viewer').capitalize()}")
    with st.expander("Change password"):
        _password_form(a)
    if st.button(label, key="mizu_logout"):
        audit("logout")
        st.session_state.clear()
        st.rerun()

_LOGO_SVG = login_fx.logo_html()   # your real logo file when one is found
