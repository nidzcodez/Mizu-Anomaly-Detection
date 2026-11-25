"""MIZU login page effects: moving-dots background, real logo loader, extra CSS."""
import base64
import json
import os
from functools import lru_cache
from pathlib import Path

import streamlit.components.v1 as components

LOGO_FILE = ""            # optional: force a file, e.g. "assets/logo.png". Empty = search the project for *logo*.
DOT_RGB = "53, 109, 193"  # dot and line colour (MIZU blue)

_ROOT = Path(__file__).parent
_MIME = {".svg": "image/svg+xml", ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}
_SKIP = {"backups_final", "venv", "env", "node_modules", "__pycache__", "site-packages"}

_FALLBACK = (
    '<svg width="52" height="52" viewBox="0 0 48 48" xmlns="http://www.w3.org/2000/svg">'
    '<rect width="48" height="48" rx="12" fill="#356dc1"/>'
    '<path d="M24 9c6.2 7.3 10.2 12.2 10.2 17.4a10.2 10.2 0 0 1-20.4 0C13.8 21.2 17.8 16.3 24 9z" fill="#fff"/>'
    '<path d="M19.6 27.4a4.6 4.6 0 0 0 8.8 0" fill="none" stroke="#356dc1" stroke-width="2.6" stroke-linecap="round"/>'
    '</svg>'
)


@lru_cache(maxsize=1)
def logo_path():
    if LOGO_FILE:
        p = _ROOT / LOGO_FILE
        return p if p.is_file() else None
    found = []
    for dirpath, dirs, files in os.walk(_ROOT):
        dirs[:] = [d for d in dirs if d not in _SKIP and not d.startswith(".")]
        for f in files:
            p = Path(dirpath) / f
            if p.suffix.lower() in _MIME and "logo" in p.stem.lower():
                try:
                    if p.stat().st_size <= 1_500_000:
                        found.append(p)
                except OSError:
                    pass
    if not found:
        return None

    def rank(p):
        n = p.stem.lower()
        return (len(p.relative_to(_ROOT).parts), "favicon" in n or "icon" in n, p.suffix.lower() != ".svg", len(n))

    return sorted(found, key=rank)[0]


@lru_cache(maxsize=1)
def logo_html():
    p = logo_path()
    if p:
        try:
            data = base64.b64encode(p.read_bytes()).decode("ascii")
            return '<img class="mg-logoimg" alt="MIZU" src="data:%s;base64,%s">' % (_MIME[p.suffix.lower()], data)
        except OSError:
            pass
    return _FALLBACK


CSS = """
@keyframes mzRise { from { opacity: 0; transform: translateY(14px); } to { opacity: 1; transform: none; } }
@keyframes mzFloat { 0%, 100% { transform: translateY(0); } 50% { transform: translateY(-4px); } }

html { background: #f8f9fc !important; }
html body, html body .stApp, html body [data-testid="stAppViewContainer"], html body [data-testid="stMain"], html body .main {
    background: transparent !important; background-image: none !important;
}

.stApp .st-key-mizu-card-info, .stApp .st-key-mizu-card-login {
    animation: mzRise .55s ease backwards;
    transition: transform .25s ease, box-shadow .25s ease;
}
.stApp .st-key-mizu-card-login { animation-delay: .12s; }
.stApp .st-key-mizu-card-info:hover, .stApp .st-key-mizu-card-login:hover {
    transform: translateY(-3px); box-shadow: 0 10px 28px rgba(53, 109, 193, .16) !important;
}

.mg-point { animation: mzRise .5s ease backwards; transition: transform .2s ease; }
.mg-point:nth-child(3) { animation-delay: .35s; }
.mg-point:nth-child(4) { animation-delay: .45s; }
.mg-point:nth-child(5) { animation-delay: .55s; }
.mg-point:hover { transform: translateX(5px); }

.mg-logoimg { width: 52px; height: 52px; object-fit: contain; display: block; }
.mg-signlogo { animation: mzFloat 4s ease-in-out infinite; }
.mg-top .mg-logoimg, .mg-top svg { transition: transform .3s ease; }
.mg-top:hover .mg-logoimg, .mg-top:hover svg { transform: rotate(-6deg) scale(1.06); }

.stApp .st-key-mizu-card-login [data-testid="stFormSubmitButton"], .stApp .st-key-mizu-card-login [data-testid="stFormSubmitButton"] > div { width: 100% !important; }
.stApp .st-key-mizu-card-login [data-testid="stFormSubmitButton"] button {
    width: 100% !important; transition: transform .15s ease, box-shadow .2s ease, background .2s ease;
}
.stApp .st-key-mizu-card-login [data-testid="stFormSubmitButton"] button:hover {
    transform: translateY(-2px); box-shadow: 0 6px 16px rgba(53, 109, 193, .35);
}
.stApp .st-key-mizu-card-login [data-testid="stFormSubmitButton"] button:active { transform: scale(.98); }

@media (prefers-reduced-motion: reduce) { .stApp * { animation: none !important; transition: none !important; } }
"""

_DOTS_JS = """
(function () {
  var W = window, D = document;
  if (W.__mizuDotsStop) { try { W.__mizuDotsStop(); } catch (e) {} }
  var c = D.createElement('canvas');
  c.id = 'mizu-dots';
  c.style.cssText = 'position:fixed;left:0;top:0;width:100%;height:100%;z-index:-1;pointer-events:none;';
  D.body.appendChild(c);
  var ctx = c.getContext('2d');
  var dpr = Math.min(W.devicePixelRatio || 1, 2);
  var w = 0, h = 0, pts = [], mx = -9999, my = -9999, raf = 0, alive = true, seen = Date.now(), frame = 0;
  var still = W.matchMedia && W.matchMedia('(prefers-reduced-motion: reduce)').matches;
  var RGB = '__RGB__', LINK = 125, REACH = 170, PUSH = 120;

  function fit() {
    w = W.innerWidth; h = W.innerHeight;
    c.width = Math.round(w * dpr); c.height = Math.round(h * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    var n = Math.max(30, Math.min(110, Math.round(w * h / 13000)));
    while (pts.length < n) {
      pts.push({x: Math.random() * w, y: Math.random() * h,
                vx: (Math.random() - .5) * .45, vy: (Math.random() - .5) * .45,
                ex: 0, ey: 0, r: 1.5 + Math.random() * 1.7});
    }
    pts.length = n;
  }
  function move(e) { mx = e.clientX; my = e.clientY; }
  function touch(e) { if (e.touches && e.touches[0]) { mx = e.touches[0].clientX; my = e.touches[0].clientY; } }
  function leave() { mx = -9999; my = -9999; }
  function stop() {
    alive = false; W.cancelAnimationFrame(raf);
    W.removeEventListener('mousemove', move); W.removeEventListener('touchmove', touch);
    W.removeEventListener('resize', fit); D.removeEventListener('mouseleave', leave);
    if (c.parentNode) { c.parentNode.removeChild(c); }
    if (W.__mizuDotsStop === stop) { W.__mizuDotsStop = null; }
  }
  W.__mizuDotsStop = stop;

  function draw() {
    ctx.clearRect(0, 0, w, h);
    var i, j, p, q, dx, dy, d, f;
    for (i = 0; i < pts.length; i++) {
      p = pts[i];
      if (!still) {
        dx = p.x - mx; dy = p.y - my; d = Math.sqrt(dx * dx + dy * dy);
        if (d < PUSH && d > 0.1) { f = (PUSH - d) / PUSH; p.ex += (dx / d) * f * 0.9; p.ey += (dy / d) * f * 0.9; }
        p.ex *= 0.93; p.ey *= 0.93;
        p.x += p.vx + p.ex; p.y += p.vy + p.ey;
        if (p.x < -10) { p.x = w + 10; } else if (p.x > w + 10) { p.x = -10; }
        if (p.y < -10) { p.y = h + 10; } else if (p.y > h + 10) { p.y = -10; }
      }
    }
    ctx.lineWidth = 1;
    for (i = 0; i < pts.length; i++) {
      p = pts[i];
      for (j = i + 1; j < pts.length; j++) {
        q = pts[j]; dx = p.x - q.x; dy = p.y - q.y; d = Math.sqrt(dx * dx + dy * dy);
        if (d < LINK) {
          ctx.strokeStyle = 'rgba(' + RGB + ',' + (0.22 * (1 - d / LINK)).toFixed(3) + ')';
          ctx.beginPath(); ctx.moveTo(p.x, p.y); ctx.lineTo(q.x, q.y); ctx.stroke();
        }
      }
      dx = p.x - mx; dy = p.y - my; d = Math.sqrt(dx * dx + dy * dy);
      if (d < REACH) {
        ctx.strokeStyle = 'rgba(' + RGB + ',' + (0.45 * (1 - d / REACH)).toFixed(3) + ')';
        ctx.beginPath(); ctx.moveTo(p.x, p.y); ctx.lineTo(mx, my); ctx.stroke();
      }
    }
    ctx.fillStyle = 'rgba(' + RGB + ',0.55)';
    for (i = 0; i < pts.length; i++) {
      p = pts[i]; ctx.beginPath(); ctx.arc(p.x, p.y, p.r, 0, 6.2832); ctx.fill();
    }
  }
  function loop() {
    if (!alive) { return; }
    frame++;
    if (frame % 20 === 0) {
      if (D.querySelector('.st-key-mizu-card-login')) { seen = Date.now(); }
      else if (Date.now() - seen > 1500) { stop(); return; }
    }
    draw();
    raf = W.requestAnimationFrame(loop);
  }
  fit();
  W.addEventListener('mousemove', move); W.addEventListener('touchmove', touch, {passive: true});
  W.addEventListener('resize', fit); D.addEventListener('mouseleave', leave);
  loop();
})();
"""


def run():
    """Autofocus the username field and start the moving-dots background."""
    js = _DOTS_JS.replace("__RGB__", DOT_RGB)
    html = (
        "<script>(function(){try{var p=window.parent,d=p.document;"
        "var i=d.querySelector('.st-key-mizu-user input');if(i){i.focus();}"
        "var s=d.createElement('script');s.textContent=" + json.dumps(js) + ";"
        "d.body.appendChild(s);d.body.removeChild(s);}catch(e){}})();</script>"
    )
    components.html(html, height=0)
