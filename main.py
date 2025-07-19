#!/usr/bin/env python3
"""
main.py — Parser MacroDroid • v5.15-dev • iunie 2025
----------------------------------------------------
Segment 1/4  –  bootstrap, import-uri, log-setup, regex-uri,
modele de date, helper-e generale.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
import time
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from functools import lru_cache
from io import StringIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ——— integrare noul motor de parser ——— #
from sms_parser import SMSParsingEngine  # NEW

# ——— opţional phonenumbers ——— #
try:
    import phonenumbers
except ModuleNotFoundError:  # fallback elegant
    phonenumbers = None

# ——— zoneinfo (Py ≥ 3.9 sau backport) ——— #
try:
    from zoneinfo import ZoneInfo
except ModuleNotFoundError:  # Python < 3.9
    from backports.zoneinfo import ZoneInfo  # type: ignore

try:
    DEFAULT_TZ = ZoneInfo("Europe/Bucharest")
except Exception:
    DEFAULT_TZ = None

TZ: Optional[ZoneInfo] = DEFAULT_TZ
MAX_BUFFER_KB = 32  # limită buffer DEBUG / bloc

# ────────── GLOBALS runtime ─────────── #
THROTTLE_DISABLED: bool = False
THROTTLE_LIMIT_MB: int = 2  # override prin --throttle
SAMPLE_LIMIT: int = 0  # primele N blocuri ignorate salvate
SAMPLE_COLLECT: List[str] = []  # conţinutul efectiv
FAIL_COUNT: int = 0  # #blocuri eşuate (în fail_debug)

# ——— engine singleton ——— #
ENGINE = SMSParsingEngine()  # NEW

# ───────────────── LOGGING ───────────────── #
LOG_FILE = Path("parser_debug.log")

log_console = logging.StreamHandler(sys.stdout)
log_console.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))
log_console.setLevel(logging.INFO)  # implicit INFO

log_file = logging.FileHandler(LOG_FILE, "w", "utf-8")
log_file.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))
log_file.setLevel(logging.DEBUG)  # fişier = DEBUG

logging.basicConfig(level=logging.DEBUG, handlers=[log_console, log_file])
log = logging.getLogger("md_parser")


# ────── handler suplimentar pentru blocuri eşuate ────── #
def _build_fail_handler(mode: str) -> Tuple[logging.Logger, Path]:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    if mode == "overwrite":
        p = Path("fail_debug.log")
        fh = logging.FileHandler(p, "w", "utf-8")
    elif mode == "append":
        p = Path("fail_debug.log")
        fh = logging.FileHandler(p, "a", "utf-8")
    else:  # rotate (default)
        p = Path(f"fail_debug-{ts}.log")
        fh = logging.FileHandler(p, "w", "utf-8")
    fh.setFormatter(logging.Formatter("%(message)s"))
    fh.setLevel(logging.DEBUG)
    lg = logging.getLogger("fail_blocks")
    lg.propagate = False
    lg.addHandler(fh)
    lg.setLevel(logging.DEBUG)
    return lg, p


# logger fail_blocks va fi instanţiat în main()

# ─────────── CODURI LOG ─────────── #
# IG33 sms_no_time_fmt      | IG34 sms_inline_fail
# IG35 call_no_time_fmt     | IG36 call_date_only
# IG37 inline_same_line     | IG38 dt_inline_extra
# IG39 call_fallback_midnight
# IG40 consolidate_forced_close
# IG41 summary_reject_reasons
# IG42 log_throttle
# IG43 sms_datafirst_ok
# IG44 sms_brief_ok
# IG45 sms_postbody_ok
# IG46 block_fail_flush
# IG47 bad_timezone
# IG48 sms_fallback_ok
# IG49 sms_skip_ignore
# ────────────────────────────────── #

# ───────── HELPER datetime cu TZ ───────── #
def _dt(y: int, m: int, d: int, h: int, mi: int, s: int = 0) -> datetime:
    return datetime(y, m, d, h, mi, s, tzinfo=TZ)


# ───────── context-manager pentru buffer ───────── #
@contextmanager
def block_trace(idx: int, raw_header: str, stats: Dict[str, int]):
    """
    Capturează DEBUG-urile unui bloc; descarcă în fail_log dacă blocul eşuează.
    Buffer-ul e trunchiat la MAX_BUFFER_KB pentru a limita memoria.
    """
    global FAIL_COUNT, fail_log
    buf = StringIO()
    h = logging.StreamHandler(buf)
    h.setLevel(logging.DEBUG)
    log.addHandler(h)

    sms_ign0 = stats.get("sms_ignore", 0)
    call_ign0 = stats.get("call_ignore", 0)
    t0 = time.perf_counter()

    try:
        yield
    finally:
        log.removeHandler(h)
        if (stats.get("sms_ignore", 0) > sms_ign0) or (
                stats.get("call_ignore", 0) > call_ign0
        ):
            FAIL_COUNT += 1
            data = buf.getvalue()
            if len(data) > MAX_BUFFER_KB * 1024:
                head = data[: 4 * 1024]
                tail = data[-4 * 1024:]
                kb = len(data) // 1024
                data = f"{head}\n...[TRUNCATED {kb - 8} KB]...\n{tail}"
            header = raw_header.replace("\n", " ")[:120]
            fail_log.debug("\n\n--- BLOCK #%d  %s\n%s", idx, header, data)
            fail_log.debug("IG46 block_fail_flush #%d Δt=%.3fs", idx, time.perf_counter() - t0)


# ───────────────── CONSTANTE ───────────────── #
_MONTHS = {m.lower(): i for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June",
     "July", "August", "September", "October", "November", "December"], 1)}
_MONTHS |= {
    "ianuarie": 1, "februarie": 2, "martie": 3, "aprilie": 4, "mai": 5, "iunie": 6,
    "iulie": 7, "august": 8, "septembrie": 9, "octombrie": 10, "noiembrie": 11, "decembrie": 12,
}

_SEPARATOR = re.compile(r"^-{10,}\s*$", re.MULTILINE)

# ▶ Blocuri CALL – regex-uri
_CALL_TIME = re.compile(
    r"^Call\s+(?P<trig>Incoming|Active|Ended|Missed|Outgoing)\b[^\n]*\n"
    r"1\.\s*Data\s+(?P<day>\d{1,2})\s+(?P<month>\w+)\s+(?P<year>\d{4})\n"
    r"2\.\s*Ora:\s*(?P<hour>\d{1,2}):(?!:)(?P<minute>\d{2})"
    r"(?:\s+(?P<ampm>AM|PM|am|pm))?"
    r"(?:\s*(?P<second>\d{1,2}))?\s*s?",
    re.MULTILINE,
)

_CALL_DATEONLY = re.compile(
    r"^Call\s+(?P<trig>Incoming|Active|Ended|Missed|Outgoing)\b[^\n]*\n"
    r"1\.\s*Data\s+(?P<day>\d{1,2})\s+(?P<month>\w+)\s+(?P<year>\d{4})",
    re.MULTILINE,
)

# ▶ Blocuri SMS – regex-uri principale
_SMS_TIME = re.compile(
    r"^SMS:.*?\n"
    r"(?:\s*(?P<number>\+?\d+)?\s*(?P<name>[^\n]*))?\n"
    r"1\.\s*Data\s+(?P<day>\d{1,2})\s+(?P<month>\w+)\s+(?P<year>\d{4})\n"
    r"2\.\s*Ora:\s*(?P<hour>\d{1,2}):(?!:)(?P<minute>\d{2})"
    r"(?:\s+(?P<ampm>AM|PM|am|pm))?"
    r"(?:\s*(?P<second>\d{1,2}))?\s*s?",
    re.MULTILINE,
)

_SMS_GENERIC = re.compile(r"^SMS:\s*(?P<header>[^\n]+?)\n(?P<body>[\s\S]+)", re.MULTILINE)

# (urmează regex-uri adiţionale, modele de date, helper-e etc. în segmentul 2)

# —— SEGMENT 2/4 ——————————————————————————————————————————
# Continuare: regex-uri suplimentare, date-models, helper-uri, parsere SMS
# ————————————————————————————————————————————————————————————

# ▶ Dată inline BT / POS etc.
_DT_INLINE = re.compile(
    r"""
    (?:data\s*(?:si|și)?\s*ora[:\s]*)?
    (?P<day>\d{1,2})
    (?:[./]|\s+)
    (?P<month>\d{1,2}|[A-Za-zăâîșţ]+)
    (?:[./]|\s+)
    (?P<year>\d{2,4})
    [\s,]*
    (?P<hour>\d{1,2})
    :
    (?P<minute>\d{2})
    (?::(?P<second>\d{1,2}))?
    (?:\s*(?P<ampm>am|pm|AM|PM))?
    (?:\s*(?P<second2>\d{1,2})\s*s?)?
""",
    re.I | re.X,
)

# ▶ Dată + oră pe aceeaşi linie
_INLINE_SAME = re.compile(
    r"""
    data\s*(?:si|și)?\s*ora[:\s]+
    (?P<day>\d{1,2})[./](?P<month>\d{1,2})[./](?P<year>\d{2,4})\s+
    (?P<hour>\d{1,2}):(?P<minute>\d{2})(?::(?P<second>\d{1,2}))?
""",
    re.I | re.X,
)

# ▶ Data/Ora reutilizabilă
_DATA_ORA = re.compile(
    r"1\.\s*Data\s+(?P<day>\d{1,2})\s+(?P<month>\w+)\s+(?P<year>\d{4})\n"
    r"2\.\s*Ora:\s*(?P<hour>\d{1,2}):(?!:)(?P<minute>\d{2})"
    r"(?:\s+(?P<ampm>AM|PM|am|pm))?"
    r"(?:\s*(?P<second>\d{1,2}))?\s*s?",
    re.MULTILINE,
)

_SMS_HEADER_LINE = re.compile(r"^SMS:\s*(?P<header>[^\n]+)", re.MULTILINE)

# ▶ SMS fără bloc Data/Ora
_SMS_BRIEF = re.compile(
    r"^SMS:.*?\n"
    r"(?:\s*(?P<number>\+?\d+)?\s*(?P<name>[^\n]*))?\n"
    r"(?P<day>\d{1,2})\s+(?P<month>\w+)\s+(?P<year>\d{4})\s+"
    r"(?P<hour>\d{1,2}):(?!:)(?P<minute>\d{2})"
    r"(?:\:(?P<second>\d{1,2}))?"
    r"(?:\s*(?P<ampm>am|pm|AM|PM))?",
    re.MULTILINE,
)


# —──────────── UTILS & DATA-MODELS —──────────── #
def _year_norm(y: str) -> int:
    return int(y) if len(y) == 4 else (2000 + int(y) if int(y) < 70 else 1900 + int(y))


_TRIG_SMS_SENT = ["sms sent to", "sms trimis la"]
_TRIG_SMS_RECV = ["sms from", "mesaj sms de la"]


@dataclass
class CallEvent:
    ts: datetime
    trigger: str
    number: str
    name: str


@dataclass
class SMSMessage:
    direction: str
    number: str
    name: str
    timestamp: datetime
    content: str


@dataclass
class ConsolidatedCall:
    direction: str
    number: str
    name: str
    initiated_ts: datetime
    answered_ts: Optional[datetime]
    ended_ts: Optional[datetime]
    outcome: str
    ring_seconds: Optional[int]


# —──────── phone-helpers —──────── #
def _m(x: str) -> int:
    return _MONTHS[x.lower()]


_iso = lambda d: d.isoformat(sep=" ") if d else ""


def detect_trigger(h: str) -> str:
    h = h.lower()
    mapping = {
        "start_in": ["incomingcalltrigger", "call incoming"],
        "start_out": ["outgoingcalltrigger", "call outgoing"],
        "active": ["callactivetrigger", "call active"],
        "ended": ["callendedtrigger", "call ended"],
        "missed": ["callmissedtrigger", "call missed"],
    }
    for k, v in mapping.items():
        if any(t in h for t in v):
            return k
    return "other"


# ▶ regex phone RO/intl
_PHONE_RE = re.compile(
    r"""
    (?:
        (?:\+|00)?4?0\s*[17]\d{2}\s*\d{3}\s*\d{3}  # +40 7xx xxx xxx
      | 07\d{8}                                    # 07xxxxxxxx
      | \d{10}                                     # 10 digits consecutive
    )
""",
    re.X,
)


@lru_cache(maxsize=2048)
def extract_phone_number(text: str) -> str:
    """Returnează primul număr valid găsit în text (E.164 sau 07xx)."""
    if phonenumbers:
        for m in phonenumbers.PhoneNumberMatcher(text, None):
            return phonenumbers.format_number(m.number, phonenumbers.PhoneNumberFormat.E164)
    m = _PHONE_RE.search(text)
    return re.sub(r"\D", "", m.group(0)) if m else ""


# —──────── colector mostre ignorate —──────── #
def _collect_ignore(raw: str):
    if SAMPLE_LIMIT == 0:
        return
    if len(SAMPLE_COLLECT) < SAMPLE_LIMIT:
        SAMPLE_COLLECT.append(raw)


# —──────── SMS PARSERS DE BAZĂ (1/3) —──────── #

def _sms_from_datafirst(raw: str, idx: int, stats: Dict[str, int]) -> Optional[SMSMessage]:
    """Caz «Data/Ora» apare înainte de header «SMS: …»."""
    m_ts = _DATA_ORA.search(raw)
    m_hd = _SMS_HEADER_LINE.search(raw)
    if not (m_ts and m_hd and m_ts.start() < m_hd.start()):
        return None

    g = m_ts.groupdict()
    h = int(g["hour"])
    minute = int(g["minute"])
    sec = int(g.get("second") or 0)
    am = (g.get("ampm") or "").lower()
    if am == "pm" and h < 12:
        h += 12
    if am == "am" and h == 12:
        h = 0
    ts = _dt(int(g["year"]), _m(g["month"]), int(g["day"]), h, minute, sec)

    header = m_hd.group("header").strip()
    body = raw[m_hd.end():m_ts.start()].strip()

    num = extract_phone_number(header) or extract_phone_number(body)
    name = header
    if num:
        pos = header.find(num)
        if pos != -1:
            name = (header[:pos] + header[pos + len(num):]).strip()

    hdr_low = header.lower()
    direction = (
        "sent" if any(t in hdr_low for t in _TRIG_SMS_SENT)
        else "received" if any(t in hdr_low for t in _TRIG_SMS_RECV)
        else ("sent" if "sent" in hdr_low else "received")
    )

    stats["sms_match"] += 1
    log.debug("IG43 sms_datafirst_ok #%d %s %s", idx, direction, num)
    return SMSMessage(direction, num, name or header, ts, body)


# (sectoarele _sms_from_brief, _sms_from_postbody şi _parse_sms continuă
#  în segmentul următor)

# —— SEGMENT 3/4 ——————————————————————————————————————————
# Continuare: alţi parsere SMS, decorator fallback, primele fallback-uri
# ————————————————————————————————————————————————————————————

def _sms_from_brief(raw: str, idx: int, stats: Dict[str, int]) -> Optional[SMSMessage]:
    """Formă scurtă: «SMS: …» urmat de dată/ora pe rândul următor."""
    m = _SMS_BRIEF.search(raw)
    if not m:
        return None
    g = m.groupdict()
    h = int(g["hour"])
    minute = int(g["minute"])
    sec = int(g.get("second") or 0)
    am = (g.get("ampm") or "").lower()
    if am == "pm" and h < 12:
        h += 12
    if am == "am" and h == 12:
        h = 0
    ts = _dt(int(g["year"]), _m(g["month"]), int(g["day"]), h, minute, sec)

    header = raw.split("\n", 1)[0][4:].strip()
    body = raw[m.end():].strip()

    num = extract_phone_number(header) or (g.get("number") or "").strip() or extract_phone_number(body)
    name = (g.get("name") or "").strip() or header
    if num:
        pos = name.find(num)
        if pos != -1:
            name = (name[:pos] + name[pos + len(num):]).strip()

    hdr_low = header.lower()
    direction = (
        "sent" if any(t in hdr_low for t in _TRIG_SMS_SENT)
        else "received" if any(t in hdr_low for t in _TRIG_SMS_RECV)
        else ("sent" if "sent" in hdr_low else "received")
    )

    stats["sms_match"] += 1
    log.debug("IG44 sms_brief_ok #%d %s %s", idx, direction, num)
    return SMSMessage(direction, num, name or header, ts, body)


def _sms_from_postbody(raw: str, idx: int, stats: Dict[str, int]) -> Optional[SMSMessage]:
    """Caz: header «SMS: …» apoi corp, iar blocul Data/Ora după corp."""
    m_ts = _DATA_ORA.search(raw)
    m_hd = _SMS_HEADER_LINE.search(raw)
    if not (m_ts and m_hd and m_ts.start() > m_hd.end()):
        return None

    g = m_ts.groupdict()
    h = int(g["hour"])
    minute = int(g["minute"])
    sec = int(g.get("second") or 0)
    am = (g.get("ampm") or "").lower()
    if am == "pm" and h < 12:
        h += 12
    if am == "am" and h == 12:
        h = 0
    ts = _dt(int(g["year"]), _m(g["month"]), int(g["day"]), h, minute, sec)

    header = m_hd.group("header").strip()
    body = raw[m_hd.end():m_ts.start()].strip()

    num = extract_phone_number(header) or extract_phone_number(body)
    name = header
    if num:
        pos = header.find(num)
        if pos != -1:
            name = (header[:pos] + header[pos + len(num):]).strip()

    hdr_low = header.lower()
    direction = (
        "sent" if any(t in hdr_low for t in _TRIG_SMS_SENT)
        else "received" if any(t in hdr_low for t in _TRIG_SMS_RECV)
        else ("sent" if "sent" in hdr_low else "received")
    )

    stats["sms_match"] += 1
    log.debug("IG45 sms_postbody_ok #%d %s %s", idx, direction, num)
    return SMSMessage(direction, num, name or header, ts, body)


# —──────── SMS PARSER principal —──────── #
def _parse_sms(raw: str, idx: int, stats: Dict[str, int]) -> Optional[SMSMessage]:
    """
    Parser principal «bloc complet».  NU incrementează sms_ignore;
    decizia finală se ia la nivelul parse_log.
    """
    m = _SMS_TIME.search(raw)
    used = "TIME"
    if not m:
        log.debug("IG33 sms_no_time_fmt #%d %.60s", idx, raw.replace("\n", " "))
        m = _SMS_GENERIC.search(raw)
        used = "GEN"

    if not m:
        log.debug("IG22 sms_no_regex #%d", idx)
        return None

    g = m.groupdict()

    # —— timestamp —— #
    if used == "TIME":
        h = int(g["hour"])
        minute = int(g["minute"])
        sec = int(g.get("second") or 0)
        am = g.get("ampm", "").lower()
        if am == "pm" and h < 12:
            h += 12
        if am == "am" and h == 12:
            h = 0
        ts = _dt(int(g["year"]), _m(g["month"]), int(g["day"]), h, minute, sec)
    else:
        body = g["body"]
        log.debug("IG30 sms_body #%d %.50s", idx, body.replace('\n', ' '))
        dt = _DT_INLINE.search(body) or _INLINE_SAME.search(body)
        if not dt:
            log.debug("IG34 sms_inline_fail #%d", idx)
            log.debug("IG49 sms_skip_ignore #%d", idx)
            return None
        if _INLINE_SAME.match(body):
            log.debug("IG37 inline_same_line #%d", idx)
        log.debug("IG31 sms_dt #%d %s", idx, dt.group(0))
        d = dt.groupdict()
        h = int(d["hour"])
        minute = int(d["minute"])
        sec = int(d.get("second") or d.get("second2") or 0)
        am = (d.get("ampm") or "").lower()
        if am == "pm" and h < 12:
            h += 12
        if am == "am" and h == 12:
            h = 0
        mn_raw = d["month"]
        mn = _m(mn_raw) if not mn_raw.isdigit() else int(mn_raw)
        ts = _dt(_year_norm(d["year"]), mn, int(d["day"]), h, minute, sec)

    # —— meta —— #
    header = (g.get("header", "").strip() if used == "GEN" else raw.split("\n", 1)[0][4:].strip())
    body_section = g.get("body", "") if used == "GEN" else raw[m.end():]

    num = extract_phone_number(header) or (g.get("number") or "").strip() or extract_phone_number(body_section)
    name = (g.get("name") or "").strip() or header
    if num:
        pos = name.find(num)
        if pos != -1:
            name = (name[:pos] + name[pos + len(num):]).strip()

    hdr_low = header.lower()
    direction = (
        "sent" if any(t in hdr_low for t in _TRIG_SMS_SENT)
        else "received" if any(t in hdr_low for t in _TRIG_SMS_RECV)
        else ("sent" if "sent" in hdr_low else "received")
    )
    content = (re.sub(r"\{call_(number|name)\}", "", raw[m.end():]).strip() if used == "TIME" else g["body"].strip())

    stats["sms_match"] += 1
    log.debug("IG23 sms_ok #%d %s %s", idx, direction, num)
    return SMSMessage(direction, num, name or header, ts, content)


# —──────── FALLBACK decorator & registru —──────── #
_SMS_FALLBACKS: List[Tuple[int, str, callable]] = []


def fallback(name: str, priority: int = 50):
    """Decoratează o funcţie fallback SMS şi o înscrie în registru."""

    def deco(fn):
        _SMS_FALLBACKS.append((priority, name, fn))
        return fn

    return deco


# —— PRIMUL set de fallback-uri (restul în segmentul 4) —— #
@fallback("fb_inline_same", 10)
def fb_inline_same(raw: str, idx: int, stats: Dict[str, int]) -> Optional[SMSMessage]:
    m_head = re.match(
        r"^SMS:\s*(?P<header>[^\n]+?)\s+"
        r"(?P<day>\d{1,2})[./](?P<month>\d{1,2})[./](?P<year>\d{2,4})\s+"
        r"(?P<hour>\d{1,2}):(?P<minute>\d{2})",
        raw,
    )
    if not m_head:
        return None
    g = m_head.groupdict()
    ts = _dt(_year_norm(g["year"]), int(g["month"]), int(g["day"]), int(g["hour"]), int(g["minute"]))
    header = g["header"].strip()
    body = raw[m_head.end():].strip()
    num = extract_phone_number(header) or extract_phone_number(body)
    direction = "received" if "from" in header.lower() else "sent"
    stats["sms_match"] += 1
    log.debug("IG48 sms_fallback_ok fb_inline_same #%d %s %s", idx, direction, num)
    return SMSMessage(direction, num, header, ts, body)


@fallback("fb_header_date", 20)
def fb_header_date(raw: str, idx: int, stats: Dict[str, int]) -> Optional[SMSMessage]:
    m = re.match(
        r"^SMS:\s*(?P<header>.+?)\s+\((?P<day>\d{1,2})\s+"
        r"(?P<month>\w+)\s+(?P<year>\d{4})\s+(?P<hour>\d{1,2}):(?P<minute>\d{2})\)",
        raw,
    )
    if not m:
        return None
    g = m.groupdict()
    ts = _dt(int(g["year"]), _m(g["month"]), int(g["day"]), int(g["hour"]), int(g["minute"]))
    header = g["header"].strip()
    body = raw[m.end():].strip()
    num = extract_phone_number(header) or extract_phone_number(body)
    direction = "received" if "from" in header.lower() else "sent"
    stats["sms_match"] += 1
    log.debug("IG48 sms_fallback_ok fb_header_date #%d %s %s", idx, direction, num)
    return SMSMessage(direction, num, header, ts, body)


# (fallback-urile rămase + parser Call, consolidate etc. în segmentul 4)

# —— SEGMENT 4/4 ——————————————————————————————————————————
# Fallback-uri suplimentare, sortare registru, parser Call, NEW consolidate
# cu regula „închide orice apel deschis când începe altul”,
# utilitare I/O, CLI şi main().
# ————————————————————————————————————————————————————————————

@fallback("fb_post_footer", 30)
def fb_post_footer(raw: str, idx: int, stats: Dict[str, int]) -> Optional[SMSMessage]:
    m_head = re.match(r"^SMS:\s*(?P<header>[^\n]+)", raw)
    m_tail = re.search(
        r"(?P<day>\d{1,2})\s+(?P<month>\w+)\s+(?P<year>\d{4})\s+"
        r"(?P<hour>\d{1,2}):(?P<minute>\d{2})\s*$",
        raw,
    )
    if not (m_head and m_tail):
        return None
    g = m_tail.groupdict()
    ts = _dt(int(g["year"]), _m(g["month"]), int(g["day"]), int(g["hour"]), int(g["minute"]))
    header = m_head.group("header").strip()
    body = raw[m_head.end():m_tail.start()].strip()
    num = extract_phone_number(header) or extract_phone_number(body)
    direction = "received" if "from" in header.lower() else "sent"
    stats["sms_match"] += 1
    log.debug("IG48 sms_fallback_ok fb_post_footer #%d %s %s", idx, direction, num)
    return SMSMessage(direction, num, header, ts, body)


@fallback("fb_data_before", 40)
def fb_data_before(raw: str, idx: int, stats: Dict[str, int]) -> Optional[SMSMessage]:
    m_dt = re.match(
        r"(?P<day>\d{1,2})\s+(?P<month>\w+)\s+(?P<year>\d{4})\s+"
        r"(?P<hour>\d{1,2}):(?P<minute>\d{2})\s+SMS:\s*(?P<header>[^\n]+)\n(?P<body>[\s\S]+)",
        raw,
    )
    if not m_dt:
        return None
    g = m_dt.groupdict()
    ts = _dt(int(g["year"]), _m(g["month"]), int(g["day"]), int(g["hour"]), int(g["minute"]))
    header = g["header"].strip()
    body = g["body"].strip()
    num = extract_phone_number(header) or extract_phone_number(body)
    direction = "received" if "from" in header.lower() else "sent"
    stats["sms_match"] += 1
    log.debug("IG48 sms_fallback_ok fb_data_before #%d %s %s", idx, direction, num)
    return SMSMessage(direction, num, header, ts, body)


@fallback("fb_bracketed", 50)
def fb_bracketed(raw: str, idx: int, stats: Dict[str, int]) -> Optional[SMSMessage]:
    if not raw.lstrip().startswith("[SMS]"):
        return None
    raw2 = "SMS:" + raw.lstrip()[5:]
    return _parse_sms(raw2, idx, stats)


@fallback("fb_multiline", 60)
def fb_multiline(raw: str, idx: int, stats: Dict[str, int]) -> Optional[SMSMessage]:
    if "•••" not in raw:
        return None
    for part in raw.split("•••"):
        res = _parse_sms(part.strip(), idx, stats)
        if res:
            return res
    return None


@fallback("fb_unicode_quotes", 70)
def fb_unicode_quotes(raw: str, idx: int, stats: Dict[str, int]) -> Optional[SMSMessage]:
    if "“" not in raw and "”" not in raw:
        return None
    return _parse_sms(raw.replace("“", "\"").replace("”", "\""), idx, stats)


@fallback("fb_missing_dot", 80)
def fb_missing_dot(raw: str, idx: int, stats: Dict[str, int]) -> Optional[SMSMessage]:
    if "\n1 Data" not in raw:
        return None
    raw2 = raw.replace("\n1 Data", "\n1. Data").replace("\n2 Ora", "\n2. Ora")
    return _parse_sms(raw2, idx, stats)


@fallback("fb_short_header", 90)
def fb_short_header(raw: str, idx: int, stats: Dict[str, int]) -> Optional[SMSMessage]:
    m = re.match(r"^SMS\s+de\s+la\s+(?P<num>\+?\d{6,})\s*:\s*(?P<body>.+)", raw, re.I)
    if not m:
        return None
    g = m.groupdict()
    ts = _dt(datetime.now().year, datetime.now().month, datetime.now().day, datetime.now().hour, datetime.now().minute)
    num = g["num"]
    body = g["body"].strip()
    stats["sms_match"] += 1
    log.debug("IG48 sms_fallback_ok fb_short_header #%d received %s", idx, num)
    return SMSMessage("received", num, "", ts, body)


@fallback("fb_corrupted", 100)
def fb_corrupted(raw: str, idx: int, stats: Dict[str, int]) -> Optional[SMSMessage]:
    m = re.search(r"(\+?07\d{8})(.*)", raw, re.S)
    if not m:
        return None
    num, body = m.groups()
    if len(body.strip()) < 5:
        return None
    ts = _dt(datetime.now().year, datetime.now().month, datetime.now().day, datetime.now().hour, datetime.now().minute)
    stats["sms_match"] += 1
    log.debug("IG48 sms_fallback_ok fb_corrupted #%d heuristic %s", idx, num)
    return SMSMessage("unknown", num, "", ts, body.strip())


# ——— finalizează registrul fallback ——— #
_SMS_FALLBACKS.sort(key=lambda t: t[0])


# —────────────────── CALL PARSER —────────────────── #
def _parse_call(raw: str, idx: int, stats: Dict[str, int]) -> Optional[CallEvent]:
    m = _CALL_TIME.search(raw)
    if not m:
        m_date = _CALL_DATEONLY.search(raw)
        if m_date:
            stats["call_no_time"] += 1
            log.debug("IG36 call_date_only #%d", idx)
            g = m_date.groupdict()
            if g["trig"].lower() in ("ended", "missed"):
                ts = _dt(int(g["year"]), _m(g["month"]), int(g["day"]), 0, 0, 0)
                log.debug("IG39 call_fallback_midnight #%d", idx)
                return CallEvent(ts, f"Call {g['trig']}", "", "")
        else:
            stats["call_ignore"] += 1
            _collect_ignore(raw)
            log.debug("IG35 call_no_time_fmt #%d", idx)
        return None

    g = m.groupdict()
    h = int(g["hour"])
    minute = int(g["minute"])
    sec = int(g.get("second") or 0)
    am = g.get("ampm", "").lower()
    if am == "pm" and h < 12:
        h += 12
    if am == "am" and h == 12:
        h = 0
    ts = _dt(int(g["year"]), _m(g["month"]), int(g["day"]), h, minute, sec)
    num = extract_phone_number(raw) or (g.get("number") or "").strip()
    name = (g.get("name") or "").strip()
    header = f"Call {g['trig']}"
    stats["call_match"] += 1
    log.debug("IG20 call_ok #%d %s %s", idx, g["trig"], num)
    return CallEvent(ts, header, num, name)


# —──────────────── NEWLINE RE-INJECT —──────────────── #
def _reinject_newlines(txt: str, stats: Dict[str, int]) -> str:
    txt, n1 = re.subn(r"(?<!\n)(SMS:)", r"\n\1", txt)
    txt, n2 = re.subn(r"(?<!\n)(Call\s+)", r"\n\1", txt)
    txt, n3 = re.subn(r"(?<!\n)1\.\s*Data", r"\n1. Data", txt)
    stats["newlines"] = n1 + n2 + n3
    return txt


# —──────────────── THROTTLE helper —──────────────── #
def _apply_throttle(txt: str):
    if THROTTLE_DISABLED:
        return
    if len(txt) <= THROTTLE_LIMIT_MB * 1_000_000:
        return
    log.debug("IG42 log_throttle enabled (~%.1f MB)", len(txt) / 1e6)
    log_console.setLevel(logging.INFO)


# —──────────────── PARSE FILE —──────────────── #
def parse_log(path: Path) -> Tuple[List[CallEvent], List[SMSMessage], Dict[str, int]]:
    stats = dict(blocks=0, newlines=0, call_match=0, call_ignore=0, call_no_time=0, sms_match=0, sms_ignore=0)
    txt = path.read_text("utf-8", "ignore")
    _apply_throttle(txt)
    txt = _reinject_newlines(txt, stats)

    blocks = _SEPARATOR.split(txt)
    stats["blocks"] = len(blocks)
    calls: List[CallEvent] = []
    sms: List[SMSMessage] = []

    for idx, raw in enumerate(blocks, 1):
        raw = raw.strip()
        if not raw:
            continue
        header_line = raw.split("\n", 1)[0] if raw else ""
        with block_trace(idx, header_line, stats):
            if raw.startswith("SMS"):
                m = (_parse_sms(raw, idx, stats)
                     or _sms_from_datafirst(raw, idx, stats)
                     or _sms_from_brief(raw, idx, stats)
                     or _sms_from_postbody(raw, idx, stats))
                if not m:
                    for _, _, fb in _SMS_FALLBACKS:
                        m = fb(raw, idx, stats)
                        if m:
                            break
                if m:
                    sms.append(m)
                else:
                    stats["sms_ignore"] += 1
                    _collect_ignore(raw)
            elif raw.startswith("Call"):
                c = _parse_call(raw, idx, stats)
                if c:
                    calls.append(c)
            elif "SMS:" in raw:
                m = (_sms_from_datafirst(raw, idx, stats)
                     or _sms_from_brief(raw, idx, stats)
                     or _sms_from_postbody(raw, idx, stats))
                if not m:
                    for _, _, fb in _SMS_FALLBACKS:
                        m = fb(raw, idx, stats)
                        if m:
                            break
                if m:
                    sms.append(m)
                else:
                    stats["sms_ignore"] += 1
                    _collect_ignore(raw)
            else:
                _collect_ignore(raw)

    log.debug("IG25 calls matched=%d ignored=%d no_time=%d", stats["call_match"], stats["call_ignore"],
              stats["call_no_time"])
    log.debug("IG26 sms   matched=%d ignored=%d", stats["sms_match"], stats["sms_ignore"])
    return calls, sms, stats


# —──────────────── CONSOLIDARE avansată —──────────────── #
_MAX_GAP = timedelta(minutes=5)


def consolidate(events: List[CallEvent]) -> Tuple[List[ConsolidatedCall], int]:
    """
    ● Închide forţat orice apel deschis când apare un nou «Call Incoming/Outgoing»
      indiferent de număr (cerinţa utilizatorului).
    ● Păstrează log IG40 consolidate_forced_close pentru toate cazurile.
    """
    events.sort(key=lambda e: e.ts)
    open_c: Dict[str, ConsolidatedCall] = {}
    result: List[ConsolidatedCall] = []
    forced = 0
    last_active: Tuple[str, datetime] | None = None

    def _close(c: ConsolidatedCall, end: datetime):
        c.ended_ts = end
        if end and not c.answered_ts:
            c.ring_seconds = int((end - c.initiated_ts).total_seconds())
        result.append(c)

    for ev in events:
        trig = detect_trigger(ev.trigger)
        if trig in ("start_in", "start_out"):
            direction = "incoming" if trig == "start_in" else "outgoing"

            # 1️⃣  Închide TOATE apelurile încă deschise, indiferent de număr
            for num, oc in list(open_c.items()):
                forced += 1
                gap = (ev.ts - oc.initiated_ts).total_seconds()
                log.debug("IG40 consolidate_forced_close %s gap=%.0fs", num or "", gap)
                _close(open_c.pop(num), ev.ts)

            # 2️⃣  Deschide noul apel
            open_c[ev.number] = ConsolidatedCall(direction, ev.number, ev.name, ev.ts, None, None, "initiated", None)

        elif trig == "active":
            if last_active == (ev.number, ev.ts):
                continue
            last_active = (ev.number, ev.ts)
            oc = open_c.get(ev.number)
            if oc and ev.ts - oc.initiated_ts <= _MAX_GAP:
                oc.answered_ts = ev.ts
                oc.ring_seconds = int((ev.ts - oc.initiated_ts).total_seconds())
                oc.outcome = f"{oc.direction}_answered"

        elif trig == "ended":
            oc = open_c.pop(ev.number, None)
            if oc:
                _close(oc, ev.ts)
            else:
                result.append(
                    ConsolidatedCall("unknown", ev.number, ev.name, ev.ts, None, ev.ts, "ended_without_context", None))

        elif trig == "missed":
            result.append(ConsolidatedCall("incoming", ev.number, ev.name, ev.ts, None, ev.ts, "missed", 0))

    # finalizează orice apel rămas deschis la finalul fişierului
    for oc in open_c.values():
        forced += 1
        _close(oc, None)
        oc.outcome = "incomplete"

    return result, forced


# —──────────────── CSV & IO utilities —──────────────── #
def write_csv(path: Path, rows: List[dict], headers: List[str], label: str):
    tmp = path.with_suffix(".tmp")
    try:
        with tmp.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, headers)
            w.writeheader()
            w.writerows(rows)
        tmp.replace(path)
        log.info("CSV %s scris (%d)", path.name, len(rows))
    except Exception as exc:
        log.error("Eroare la scrierea CSV %s: %s", path.name, exc)
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        raise


def discover(recursive: bool) -> List[Path]:
    pat = re.compile(r"(call|sms)", re.I)
    it = Path.cwd().rglob("*") if recursive else Path.cwd().iterdir()
    return sorted(p for p in it if p.is_file() and pat.search(p.name))


# —──────────────── CLI —──────────────── #
def cli() -> argparse.Namespace:
    P = argparse.ArgumentParser("Parser MacroDroid")
    P.add_argument("logfiles", nargs="*", type=Path)
    P.add_argument("--recursive", action="store_true")
    P.add_argument("--out", default="combined")
    P.add_argument("--debug", action="store_true")
    P.add_argument("--no-throttle", action="store_true")
    P.add_argument("--throttle", type=int, metavar="MB")
    P.add_argument("--sample-errors", type=int, metavar="N", help="Salvează primele N blocuri ignorate")
    P.add_argument("--log-console-level", choices=["INFO", "DEBUG"])
    P.add_argument("--fail-log-mode", choices=["overwrite", "append", "rotate"], default="rotate")
    P.add_argument("--tz", metavar="ZONE", help="Timezone IANA (ex. Europe/Chisinau)")
    return P.parse_args()


# —──────────────── MAIN —──────────────── #
def main():
    global THROTTLE_DISABLED, THROTTLE_LIMIT_MB, SAMPLE_LIMIT, fail_log, TZ
    a = cli()

    # TZ
    if a.tz:
        try:
            TZ = ZoneInfo(a.tz)
        except Exception as e:
            TZ = DEFAULT_TZ
            log.error("IG47 bad_timezone '%s' -> %s", a.tz, e)

    fail_log, fail_path = _build_fail_handler(a.fail_log_mode)

    if a.log_console_level:
        log_console.setLevel(getattr(logging, a.log_console_level))
    if a.debug:
        log_console.setLevel(logging.DEBUG)

    THROTTLE_DISABLED = a.no_throttle
    if a.throttle:
        THROTTLE_LIMIT_MB = a.throttle
    SAMPLE_LIMIT = 100 if a.sample_errors is None else max(0, a.sample_errors)

    files = a.logfiles or discover(a.recursive)
    if not files:
        log.error("No input files")
        sys.exit(1)

    totals = Counter()
    rej_counter = Counter()
    calls_raw: List[CallEvent] = []
    sms_raw: List[SMSMessage] = []

    for fp in files:
        c, s, st = parse_log(fp)
        calls_raw += c
        sms_raw += s
        totals.update(st)
        rej_counter.update(
            {"call_no_time": st["call_no_time"], "call_ignore": st["call_ignore"], "sms_ignore": st["sms_ignore"]})

    calls_cons, forced = consolidate(calls_raw)
    totals["fail_blocks"] = FAIL_COUNT

    # — CSV output — #
    prefix = a.out
    write_csv(
        Path(f"{prefix}_calls.csv"),
        [asdict(c) | {
            "transport": "call",
            "initiated_ts": _iso(c.initiated_ts),
            "answered_ts": _iso(c.answered_ts),
            "ended_ts": _iso(c.ended_ts),
            "ring_seconds": c.ring_seconds or "",
        } for c in calls_cons],
        ["transport", "number", "name", "direction", "outcome", "initiated_ts", "answered_ts", "ended_ts",
         "ring_seconds"],
        "calls",
    )

    def _row_sms(m: SMSMessage) -> dict:
        p = ENGINE.parse(m.content)
        return {
            "transport": "sms",
            "number": m.number,
            "name": m.name,
            "direction": m.direction,
            "timestamp": _iso(m.timestamp),
            "content_len": len(m.content),
            "content": m.content[:256],
            "p_name": p.get("name", "") if p else "",
            "p_amount": p.get("amount", "") if p else "",
            "p_date": p.get("date", "") if p else "",
        }

    write_csv(
        Path(f"{prefix}_sms.csv"),
        [_row_sms(m) for m in sms_raw],
        ["transport", "number", "name", "direction", "timestamp", "content_len", "content", "p_name", "p_amount",
         "p_date"],
        "sms",
    )

    Path("parser_stats.json").write_text(json.dumps({
        "totals": totals,
        "consolidated_calls": len(calls_cons),
        "forced_closes": forced,
        "reject_top5": rej_counter.most_common(5),
        "fail_log": str(fail_path),
        "timezone": str(TZ) if TZ else "naive",
    }, indent=2, ensure_ascii=False), "utf-8")

    if SAMPLE_LIMIT and SAMPLE_COLLECT:
        Path("ignored_samples.txt").write_text("\n\n-----\n\n".join(SAMPLE_COLLECT), "utf-8")
        log.info("Mostre ignorate salvate: %d / %d", len(SAMPLE_COLLECT), SAMPLE_LIMIT)

    log.info("✔ OK. CSV-urile şi parser_stats.json au fost generate. Detalii: %s", LOG_FILE.name)
    log.info("   Blocuri eşuate descărcate: %d → %s", FAIL_COUNT, fail_path.name)
    log.debug("IG00 sms_parser_loaded %s", ENGINE.__class__.__name__)


if __name__ == "__main__":
    main()