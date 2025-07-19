#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sms_parser.py – Enhanced SMS Parser with broad-format support
Author: Cod Py 2 team + Fixes by Assistant
"""

from __future__ import annotations

import json
import logging
import re
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import datetime
from functools import wraps
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Callable, List, Optional

# ----------------------------- Logging Setup -----------------------------
LOG_DIR = Path(__file__).with_suffix("").parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

logger = logging.getLogger("sms_parser")
logger.setLevel(logging.DEBUG)

_file_handler = RotatingFileHandler(
    LOG_DIR / "parser_debug.log", maxBytes=2 * 1024 * 1024, backupCount=3, encoding="utf-8"
)
_file_handler.setFormatter(
    logging.Formatter(fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
)
logger.addHandler(_file_handler)

# ----------------------------- Decorator -----------------------------
def instrument(fn: Callable) -> Callable:
    parser_name = fn.__qualname__

    @wraps(fn)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            result = fn(*args, **kwargs)
            elapsed = (time.perf_counter() - start) * 1000
            if result is not None:
                logger.info(f"[{parser_name}] SUCCESS in {elapsed:.1f} ms – keys: {list(result.keys())}")
            else:
                logger.debug(f"[{parser_name}] failed in {elapsed:.1f} ms")
            return result
        except Exception as exc:
            elapsed = (time.perf_counter() - start) * 1000
            logger.exception(f"[{parser_name}] EXCEPTION after {elapsed:.1f} ms: {exc!r}")
            return None

    return wrapper

# ----------------------------- Abstract Base -----------------------------
class BaseParser(ABC):
    @abstractmethod
    @instrument
    def parse(self, text: str) -> Optional[dict]:
        pass

# ----------------------------- Parsers -----------------------------
class StandardParser(BaseParser):
    _pattern = re.compile(
        r"^(?P<name>[A-ZĂÂÎȘȚa-zăâîșț ]+):\s*(?P<amount>\d+(?:[.,]\d{1,2})?)\s*:\s*"
        r"(?P<date>\d{2}[-/]\d{2}[-/]\d{4})$"
    )

    @instrument
    def parse(self, text: str) -> Optional[dict]:
        if m := self._pattern.match(text.strip()):
            return {
                "name": m.group("name").title(),
                "amount": float(m.group("amount").replace(",", ".")),
                "date": datetime.strptime(m.group("date"), "%d-%m-%Y").date().isoformat(),
            }
        return None

class FlexibleDelimiterParser(BaseParser):
    _pattern = re.compile(
        r"^(?P<name>[A-Za-z ĂÂÎâăîșȘţȚț]+)[,;/]\s*(?P<amount>\d+(?:[.,]\d{1,2})?)"
        r"[,;/]\s*(?P<date>\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})$"
    )

    @instrument
    def parse(self, text: str) -> Optional[dict]:
        if m := self._pattern.match(text.strip()):
            day, month, year = re.split(r"[-/.]", m.group("date"))
            year = year.zfill(4)
            return {
                "name": m.group("name").title(),
                "amount": float(m.group("amount").replace(",", ".")),
                "date": f"{year}-{month.zfill(2)}-{day.zfill(2)}",
            }
        return None

class KeywordBasedParser(BaseParser):
    _kmap = {"nume": "name", "sumă": "amount", "suma": "amount", "data": "date"}
    _pair_re = re.compile(r"(\b(?:nume|sum[ăa]|data)\b)\s*[:=]\s*([\w./-]+)", re.IGNORECASE)

    @instrument
    def parse(self, text: str) -> Optional[dict]:
        out = {}
        for key, val in self._pair_re.findall(text):
            out[self._kmap[key.lower()]] = val
        if not out:
            return None
        try:
            if "amount" in out:
                out["amount"] = float(out["amount"].replace(",", "."))
            if "date" in out:
                try:
                    out["date"] = datetime.strptime(out["date"], "%d/%m/%Y").date().isoformat()
                except ValueError:
                    out["date"] = datetime.strptime(out["date"], "%d-%m-%Y").date().isoformat()
        except Exception as e:
            logger.warning(f"KeywordBasedParser conversion failed: {e}")
            return None
        return out if {"name", "amount", "date"} <= out.keys() else None

class LoosePatternParser(BaseParser):
    """Catches loose messages like 'Ali, 23.5' or 'Ali: 23.5' without date."""

    _pattern = re.compile(
        r"^(?P<name>[A-Za-zăâîșț ĂÂÎȘȚ]+)[,:]\s*(?P<amount>\d+(?:[.,]\d{1,2})?)"
    )

    @instrument
    def parse(self, text: str) -> Optional[dict]:
        if m := self._pattern.match(text.strip()):
            try:
                today = datetime.today().date().isoformat()
                return {
                    "name": m.group("name").title(),
                    "amount": float(m.group("amount").replace(",", ".")),
                    "date": today,
                }
            except Exception as e:
                logger.debug(f"LoosePatternParser error: {e}")
        return None

# ----------------------------- Engine -----------------------------
class SMSParsingEngine:
    _stats_file = LOG_DIR / "parser_stats.json"

    def __init__(self, strategies: Optional[List[BaseParser]] = None):
        self.strategies = strategies or [
            StandardParser(),
            FlexibleDelimiterParser(),
            KeywordBasedParser(),
            LoosePatternParser(),
        ]
        self._stats = defaultdict(int)
        self._load_stats()

    def parse(self, text: str) -> Optional[dict]:
        for strat in self.strategies:
            result = strat.parse(text)
            self._stats[strat.__class__.__name__] += int(result is not None)
            if result:
                self._stats["parsed_total"] += 1
                self._save_stats_async()
                return result
        logger.warning("All strategies failed; UNPARSED SMS: %s", text)
        self._stats["unparsed_total"] += 1
        self._append_ignored_sample(text)
        self._save_stats_async()
        return None

    def _load_stats(self):
        if self._stats_file.exists():
            try:
                self._stats.update(json.loads(self._stats_file.read_text(encoding="utf-8")))
            except Exception as exc:
                logger.error("Cannot read parser_stats.json: %s", exc)

    def _save_stats_async(self):
        try:
            self._stats_file.write_text(json.dumps(self._stats, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:
            logger.error("Error writing parser_stats.json: %s", exc)

    @staticmethod
    def _append_ignored_sample(text: str):
        try:
            with (LOG_DIR / "ignored_samples.txt").open("a", encoding="utf-8") as fh:
                fh.write(text.replace("\n", " ") + "\n")
        except Exception as exc:
            logger.error("Cannot write to ignored_samples.txt: %s", exc)

# ----------------------------- Public API -----------------------------
_GLOBAL_ENGINE: Optional[SMSParsingEngine] = None

def get_engine() -> SMSParsingEngine:
    global _GLOBAL_ENGINE
    if _GLOBAL_ENGINE is None:
        _GLOBAL_ENGINE = SMSParsingEngine()
    return _GLOBAL_ENGINE

def parse_sms(text: str) -> Optional[dict]:
    return get_engine().parse(text)

__all__ = ["SMSParsingEngine", "parse_sms", "get_engine"]

# ----------------------------- Quick Test -----------------------------
if __name__ == "__main__":
    for sample in [
        "Ion Pop: 123.50 : 01-06-2025",
        "Maria, 99,99, 31/05/25",
        "Nume=Alex Sumă=77.40 Data=02/06/2025",
        "Ali, 23.5",
        "Mesaj neparsabil complet"
    ]:
        print(sample, "→", parse_sms(sample))
