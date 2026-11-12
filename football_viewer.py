#!/usr/bin/env python3
"""
Football Team Viewer
====================
Chiede via CLI il nome di una squadra (non case-sensitive, accenti ignorati) e
apre una finestra PySide6 con:

  1. Mappa dei punti da cui e' stato subito un gol
  2. Tabella ordinabile per giocatore (minuti medi, tocchi medi, rapporto)
  3. Heatmap

Sorgenti dati (--source):
  pitchapi   (default) https://pitchapi.dev - gratuita, stagione CORRENTE, tutte
             le competizioni (campionati, coppe nazionali, coppe UEFA...).
             Serve una API key gratuita: https://pitchapi.dev/get-api-key
             Limite: la heatmap e' una griglia 16x12 di TUTTE le azioni con la
             palla (la fonte non espone le coordinate dei singoli passaggi).
  statsbomb  open data StatsBomb da GitHub - solo stagioni storiche, ma con
             le coordinate dei singoli passaggi (heatmap molto piu' fine).

Dipendenze:
    pip install PySide6 matplotlib mplsoccer numpy scipy requests
"""

from __future__ import annotations

import argparse
import difflib
import getpass
import json
import os
import re
import sys
import time
import unicodedata
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import requests

CACHE_DIR = Path.home() / ".cache" / "football_viewer"
CONFIG_DIR = Path.home() / ".config" / "football_viewer"
DAY = 24 * 3600
MAX_WORKERS = 6

# Heatmap StatsBomb: campo 120x80 yard -> celle da 2x2 yard (prima erano 5x5)
SB_HEAT_BINS = (60, 40)
SB_HEAT_SIGMA = 1.3


# --------------------------------------------------------------------------
# Strutture dati comuni
# --------------------------------------------------------------------------
@dataclass
class TeamData:
    team: str
    source: str
    season_label: str
    competitions: list[str]
    n_matches: int = 0
    pitch_kind: str = "custom"                    # "custom" (105x68 m) | "statsbomb"
    conceded: list[dict] = field(default_factory=list)   # x, y, xg, penalty, minute
    players: list[dict] = field(default_factory=list)    # name, matches, avg_min, avg_touches, ratio
    passes_xy: np.ndarray | None = None           # statsbomb: passaggi singoli
    heat_grid: np.ndarray | None = None           # pitchapi: griglia [x, y]
    heat_title: str = ""
    notes: list[str] = field(default_factory=list)


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c)).casefold().strip()


def progress(prefix: str, done: int, total: int) -> None:
    print(f"\r{prefix} {done}/{total}", end="", flush=True)
    if done == total:
        print()


def resolve_team(query: str, names: dict[str, str]) -> str | None:
    """names: nome normalizzato -> nome originale."""
    q = norm(query)
    if q in names:
        return names[q]
    contains = [n for k, n in names.items() if q and q in k]
    if len(contains) == 1:
        print(f"  -> uso '{contains[0]}'")
        return contains[0]
    if contains:
        print("  Piu' squadre corrispondono, specifica meglio:")
        for n in sorted(contains)[:15]:
            print(f"    - {n}")
        return None
    close = difflib.get_close_matches(q, list(names), n=5, cutoff=0.6)
    print("  Squadra non trovata." + (" Forse intendevi:" if close else ""))
    for k in close:
        print(f"    - {names[k]}")
    return None


def finalize_players(agg: dict[int | str, dict]) -> list[dict]:
    rows = []
    for a in agg.values():
        if a["matches"] == 0 or a["minutes"] <= 0:
            continue
        avg_min = a["minutes"] / a["matches"]
        avg_t = a["touches"] / a["t_matches"] if a["t_matches"] else None
        ratio = a["touches"] / a["t_minutes"] if a["t_minutes"] else None
        rows.append({"name": a["name"], "matches": a["matches"], "avg_min": avg_min,
                     "avg_touches": avg_t, "ratio": ratio})
    return rows


def new_agg() -> dict:
    return defaultdict(lambda: {"name": "", "matches": 0, "minutes": 0.0,
                                "t_matches": 0, "t_minutes": 0.0, "touches": 0})


# ==========================================================================
# SORGENTE 1: PitchAPI (stagione corrente, tutte le competizioni)
# ==========================================================================
PITCHAPI_BASE = "https://api.pitchapi.dev"


class PitchAPIError(RuntimeError):
    pass


class PitchAPI:
    def __init__(self, key: str):
        self.s = requests.Session()
        self.s.headers["X-API-KEY"] = key

    def get(self, path: str, params: dict | None = None, max_age: float | None = None):
        """GET con cache su disco e retry. Ritorna 'data' oppure None se 404."""
        sub = CACHE_DIR / "pitchapi"
        sub.mkdir(parents=True, exist_ok=True)
        tag = path.strip("/").replace("/", "__")
        if params:
            tag += "__" + "_".join(f"{k}-{v}".replace("/", "-") for k, v in sorted(params.items()))
        cache = sub / (tag + ".json")
        if cache.exists() and (max_age is None or time.time() - cache.stat().st_mtime < max_age):
            try:
                return json.loads(cache.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                cache.unlink(missing_ok=True)

        for attempt in range(5):
            try:
                r = self.s.get(PITCHAPI_BASE + path, params=params, timeout=60)
            except requests.RequestException as err:
                time.sleep(1.5 * (attempt + 1))
                last = str(err)
                continue
            if r.status_code == 401:
                raise PitchAPIError("API key non valida o mancante (401).")
            if r.status_code == 404:
                return None
            if r.status_code == 429:
                time.sleep(float(r.headers.get("Retry-After", 2)))
                last = "429"
                continue
            if r.status_code >= 500:
                time.sleep(1.5 * (attempt + 1))
                last = str(r.status_code)
                continue
            r.raise_for_status()
            data = r.json().get("data")
            cache.write_text(json.dumps(data), encoding="utf-8")
            return data
        raise PitchAPIError(f"Richiesta fallita per {path}: {last}")


def get_pitchapi_key() -> str:
    key = os.environ.get("PITCHAPI_KEY", "").strip()
    if key:
        return key
    kf = CONFIG_DIR / "pitchapi_key"
    if kf.exists():
        return kf.read_text().strip()
    print("\nServe una API key gratuita di PitchAPI (nessuna carta di credito):")
    print("  1) registrati su https://pitchapi.dev/get-api-key")
    print("  2) incolla qui la chiave (oppure impostala nella variabile PITCHAPI_KEY)")
    key = getpass.getpass("API key: ").strip()
    if not key:
        sys.exit("Nessuna chiave inserita.")
    if input("Salvarla in ~/.config/football_viewer? [s/N] ").strip().lower() == "s":
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        kf.write_text(key)
        kf.chmod(0o600)
    return key


def current_season_string(seasons: list[str], today: date | None = None) -> str | None:
    """Sceglie la stagione 'attuale' tra quelle di una lega.
    'AAAA/BBBB' (autunno-primavera): inizia in luglio -> AAAA == anno di inizio stagione.
    'AAAA' (anno solare): AAAA == anno corrente."""
    today = today or date.today()
    start_year = today.year if today.month >= 7 else today.year - 1
    for s in seasons:
        m = re.fullmatch(r"(\d{4})/(\d{4})", s.strip())
        if m and int(m.group(1)) == start_year:
            return s
        if re.fullmatch(r"\d{4}", s.strip()) and int(s) == today.year:
            return s
    return None


def pitchapi_build_index(client: PitchAPI):
    """Tutte le partite giocate della stagione corrente in tutte le competizioni."""
    leagues = (client.get("/v1/leagues", max_age=DAY) or {}).get("leagues") or []
    print(f"Indicizzo {len(leagues)} competizioni (stagione corrente)...")

    def job(lg):
        season = current_season_string(lg.get("seasons") or [])
        if not season:
            return []
        data = client.get(f"/v1/leagues/{lg['id']}/matches", {"season": season}, max_age=2 * 3600)
        if not data:
            return []
        out = []
        for m in (data.get("matches") or []):
            out.append({"id": m["id"], "date": m.get("date"),
                        "home": m["home_team"], "away": m["away_team"],
                        "league": lg["name"], "season": season,
                        "country": (lg.get("country_code") or "").upper()})
        return out

    matches: list[dict] = []
    with ThreadPoolExecutor(MAX_WORKERS) as ex:
        futures = [ex.submit(job, lg) for lg in leagues]
        for i, f in enumerate(as_completed(futures), 1):
            try:
                matches.extend(f.result())
            except PitchAPIError as err:
                print(f"\n  ! {err}")
            progress("  competizioni", i, len(futures))
    return matches, build_candidates(matches)


# Codici paese di competizioni "internazionali" (non indicano la nazionalita' del club)
INTERNATIONAL_CODES = {"", "INT", "INTL", "EUR", "EU", "WLD", "WORLD", "UEFA", "CONMEBOL",
                       "CAF", "AFC", "CONCACAF", "OFC", "XWW", "XEU", "XSA", "XNA", "XAF", "XAS"}


def build_candidates(matches: list[dict]) -> dict[str, list[dict]]:
    """nome squadra -> lista di squadre DISTINTE (per id) con quel nome.
    La nazionalita' e' quella della competizione in cui la squadra gioca piu' partite
    (escluse le competizioni internazionali), cioe' di norma il campionato nazionale."""
    stats: dict[str, dict] = {}
    for m in matches:
        for side in ("home", "away"):
            t = m[side]
            e = stats.setdefault(t["id"], {"id": t["id"], "name": t["name"], "by_league": defaultdict(int)})
            e["by_league"][(m["league"], m.get("country", ""))] += 1
    by_name: dict[str, list[dict]] = defaultdict(list)
    for e in stats.values():
        leagues = e["by_league"]
        domestic = {k: v for k, v in leagues.items() if k[1] not in INTERNATIONAL_CODES}
        (lg, country) = max((domestic or leagues).items(), key=lambda kv: kv[1])[0]
        by_name[e["name"]].append({"id": e["id"], "name": e["name"],
                                   "country": country or "?", "league": lg})
    return by_name


def choose_candidate(name: str, cands: list[dict]) -> dict:
    """Se piu' squadre hanno lo stesso nome chiede la nazionalita'."""
    if len(cands) == 1:
        return cands[0]
    cands = sorted(cands, key=lambda c: (c["country"], c["league"]))
    print(f"\nPiu' squadre si chiamano '{name}':")
    for i, c in enumerate(cands, 1):
        print(f"  {i}) {c['name']} - {c['country']} ({c['league']})")
    while True:
        a = input("Quale intendi? Inserisci il numero oppure il codice nazione: ").strip()
        if a.isdigit() and 1 <= int(a) <= len(cands):
            return cands[int(a) - 1]
        hits = [c for c in cands if norm(c["country"]) == norm(a) or norm(c["league"]) == norm(a)]
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            print("  Anche cosi' e' ambiguo: usa il numero.")
        else:
            print("  Scelta non valida.")


def _flat_stats(item: dict) -> dict:
    flat = {}
    for group in (item.get("stats") or []):
        for st in (group.get("stats") or {}).values():
            v = st.get("stat", {}).get("value")
            if v is not None:
                flat[st.get("key")] = v
    return flat


def pitchapi_process_match(client: PitchAPI, m: dict, ids: set[str]) -> dict | None:
    mid = m["id"]
    our = m["home"]["id"] if m["home"]["id"] in ids else m["away"]["id"]
    res = {"conceded": [], "players": {}, "grid": None}

    shots = client.get(f"/v1/matches/{mid}/shots")
    if shots:
        for period in (shots.get("periods") or []):
            for s in (period.get("shots") or []):
                if s.get("x") is None or s.get("y") is None:
                    continue
                if s.get("team_id") != our and s.get("event_type") == "Goal" and not s.get("is_own_goal"):
                    res["conceded"].append({
                        "x": s["x"], "y": s["y"], "xg": s.get("expected_goals"),
                        "penalty": s.get("situation") == "Penalty", "minute": s.get("minute"),
                    })

    players = client.get(f"/v1/matches/{mid}/players") or []
    adv_minutes = None
    for item in players:
        if item.get("team_id") != our:
            continue
        flat = _flat_stats(item)
        minutes = flat.get("minutes_played")
        if minutes is None:   # fallback: endpoint advanced
            if adv_minutes is None:
                adv = client.get(f"/v1/matches/{mid}/advanced/players") or {}
                adv_list = adv.get("players") if isinstance(adv, dict) else adv
                adv_minutes = {p["player"]["id"]: p.get("minutes_played")
                               for p in (adv_list or []) if isinstance(p, dict) and p.get("player")}
            minutes = adv_minutes.get(item["player"]["id"])
        if not minutes:
            continue
        res["players"][item["player"]["id"]] = {
            "name": item["player"].get("name", "?"), "minutes": float(minutes),
            "touches": flat.get("touches"),
        }

    heat = client.get(f"/v1/matches/{mid}/heatmaps")
    if heat:
        g = heat.get("grid") or {}
        L, W = g.get("length", 16), g.get("width", 12)
        grid = np.zeros((L, W))
        for t in (heat.get("teams") or []):
            if (t.get("team") or {}).get("id") == our:
                for cx, cy, n in (t.get("cells") or []):
                    if 0 <= cx < L and 0 <= cy < W:
                        grid[cx, cy] += n
        res["grid"] = grid
    return res


def collect_pitchapi(client: PitchAPI, team: str, ids: set[str], index: list[dict]) -> TeamData:
    mine = [m for m in index if m["home"]["id"] in ids or m["away"]["id"] in ids]
    mine.sort(key=lambda m: m["date"] or "")
    comps = sorted({m["league"] for m in mine})
    d = TeamData(team, "PitchAPI", mine[0]["season"] if mine else "-", comps)
    d.pitch_kind = "custom"
    if not mine:
        return d

    print(f"Scarico ed elaboro {len(mine)} partite ({len(comps)} competizioni)...")
    results = []
    with ThreadPoolExecutor(MAX_WORKERS) as ex:
        futures = [ex.submit(pitchapi_process_match, client, m, ids) for m in mine]
        for i, f in enumerate(as_completed(futures), 1):
            try:
                r = f.result()
                if r:
                    results.append(r)
            except PitchAPIError as err:
                print(f"\n  ! {err}")
            except Exception as err:   # una partita con dati anomali non blocca il resto
                print(f"\n  ! partita saltata ({type(err).__name__}: {err})")
            progress("  partite", i, len(futures))

    d.n_matches = len(results)
    agg = new_agg()
    total_grid = None
    for r in results:
        d.conceded.extend(r["conceded"])
        if r["grid"] is not None:
            total_grid = r["grid"] if total_grid is None else (total_grid + r["grid"] if total_grid.shape == r["grid"].shape else total_grid)
        for pid, p in r["players"].items():
            a = agg[pid]
            a["name"] = p["name"]
            a["matches"] += 1
            a["minutes"] += p["minutes"]
            if p["touches"] is not None:
                a["t_matches"] += 1
                a["t_minutes"] += p["minutes"]
                a["touches"] += p["touches"]
    d.players = finalize_players(agg)
    d.heat_grid = total_grid
    d.heat_title = "Heatmap azioni con la palla (griglia nativa 16x12)"
    d.notes.append("Heatmap: la fonte non fornisce le coordinate dei singoli passaggi.")
    d.notes.append("Portieri: tocchi non disponibili nella fonte.")
    return d


# ==========================================================================
# SORGENTE 2: StatsBomb open data (storico, con passaggi singoli)
# ==========================================================================
SB_BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data/"
TOUCH_TYPES = {
    "Pass", "Ball Receipt*", "Shot", "Dribble", "Miscontrol", "Dispossessed",
    "Clearance", "Ball Recovery", "Interception", "Goal Keeper",
}
RED_CARDS = {"Red Card", "Second Yellow"}


def sb_fetch(rel_path: str, max_age: float | None = None):
    sub = CACHE_DIR / "statsbomb"
    sub.mkdir(parents=True, exist_ok=True)
    cache = sub / rel_path.replace("/", "__")
    if cache.exists() and (max_age is None or time.time() - cache.stat().st_mtime < max_age):
        try:
            return json.loads(cache.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            cache.unlink(missing_ok=True)
    last = None
    for attempt in range(3):
        try:
            r = requests.get(SB_BASE + rel_path, timeout=60)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            cache.write_bytes(r.content)
            return r.json()
        except requests.RequestException as err:
            last = err
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Download fallito per {rel_path}: {last}")


def sb_season_key(name: str) -> int:
    years = re.findall(r"\d{4}", name)
    return max(map(int, years)) if years else 0


def sb_build_index() -> list[dict]:
    comps = sb_fetch("competitions.json", DAY)
    pairs = sorted({(c["competition_id"], c["season_id"]) for c in comps})
    matches: list[dict] = []
    print("Indicizzo le competizioni StatsBomb...")
    with ThreadPoolExecutor(8) as ex:
        futures = [ex.submit(lambda p: sb_fetch(f"matches/{p[0]}/{p[1]}.json", DAY) or [], p) for p in pairs]
        for i, f in enumerate(as_completed(futures), 1):
            try:
                matches.extend(f.result())
            except RuntimeError as err:
                print(f"\n  ! {err}")
            progress("  competizioni", i, len(futures))
    return matches


def sb_select(matches: list[dict], team: str) -> list[dict]:
    mine = [m for m in matches
            if team in (m["home_team"]["home_team_name"], m["away_team"]["away_team_name"])
            and m.get("match_status", "available") == "available"]
    if not mine:
        return []
    latest = max(sb_season_key(m["season"]["season_name"]) for m in mine)
    return [m for m in mine if sb_season_key(m["season"]["season_name"]) == latest]


def _t(e: dict) -> float:
    return e["minute"] + e.get("second", 0) / 60.0


def sb_process_match(events: list[dict], team: str) -> dict:
    regular = [e for e in events if e.get("period", 1) <= 4]
    match_end = max((_t(e) for e in regular), default=90.0)
    conceded, passes = [], []
    start: dict[int, float] = {}
    end: dict[int, float] = {}
    names: dict[int, str] = {}
    touches: dict[int, int] = defaultdict(int)

    for e in regular:
        et = e["type"]["name"]
        ours = e["team"]["name"] == team
        if et == "Starting XI" and ours:
            for p in e["tactics"]["lineup"]:
                start[p["player"]["id"]] = 0.0
                names[p["player"]["id"]] = p["player"]["name"]
            continue
        if et == "Shot" and not ours and e["shot"]["outcome"]["name"] == "Goal":
            x, y = e["location"][:2]
            conceded.append({"x": x, "y": y, "xg": e["shot"].get("statsbomb_xg"),
                             "penalty": e["shot"]["type"]["name"] == "Penalty", "minute": e["minute"]})
        if not ours:
            continue
        if et == "Pass" and "location" in e:
            passes.append(e["location"][:2])
        if et == "Substitution":
            out_id, rep = e["player"]["id"], e["substitution"]["replacement"]
            end[out_id] = _t(e)
            start[rep["id"]] = _t(e)
            names[rep["id"]] = rep["name"]
            names.setdefault(out_id, e["player"]["name"])
        if et in ("Foul Committed", "Bad Behaviour"):
            card = e.get(et.lower().replace(" ", "_"), {}).get("card", {}).get("name")
            if card in RED_CARDS and "player" in e:
                end[e["player"]["id"]] = _t(e)
        if "player" in e and et in TOUCH_TYPES:
            if et == "Ball Receipt*" and "outcome" in e.get("ball_receipt", {}):
                continue
            touches[e["player"]["id"]] += 1
            names.setdefault(e["player"]["id"], e["player"]["name"])

    players = {pid: {"name": names[pid], "minutes": max(0.0, end.get(pid, match_end) - s),
                     "touches": touches.get(pid, 0)} for pid, s in start.items()}
    return {"conceded": conceded, "passes": passes, "players": players}


def collect_statsbomb(team: str, matches: list[dict]) -> TeamData:
    comps = sorted({m["competition"]["competition_name"] for m in matches})
    seasons = sorted({m["season"]["season_name"] for m in matches})
    d = TeamData(team, "StatsBomb open data", " / ".join(seasons), comps)
    d.pitch_kind = "statsbomb"

    def job(m):
        ev = sb_fetch(f"events/{m['match_id']}.json")
        return None if ev is None else sb_process_match(ev, team)

    results = []
    print(f"Scarico ed elaboro {len(matches)} partite...")
    with ThreadPoolExecutor(8) as ex:
        futures = [ex.submit(job, m) for m in matches]
        for i, f in enumerate(as_completed(futures), 1):
            try:
                r = f.result()
                if r:
                    results.append(r)
            except RuntimeError as err:
                print(f"\n  ! {err}")
            progress("  partite", i, len(futures))

    d.n_matches = len(results)
    agg = new_agg()
    allp = []
    for r in results:
        d.conceded.extend(r["conceded"])
        allp.extend(r["passes"])
        for pid, p in r["players"].items():
            a = agg[pid]
            a["name"] = p["name"]
            if p["minutes"] > 0:
                a["matches"] += 1
                a["t_matches"] += 1
                a["t_minutes"] += p["minutes"]
            a["minutes"] += p["minutes"]
            a["touches"] += p["touches"]
    d.players = finalize_players(agg)
    d.passes_xy = np.array(allp, dtype=float).reshape(-1, 2)
    d.heat_title = f"Heatmap passaggi ({len(d.passes_xy)}) - celle 2x2 yard"
    d.notes.append("Sorgente storica: le stagioni StatsBomb open data non includono quella corrente.")
    return d


# ==========================================================================
# GUI (PySide6 + matplotlib/mplsoccer)
# ==========================================================================
def run_gui(data: TeamData) -> None:
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QColor, QFont, QPalette
    from PySide6.QtWidgets import (
        QAbstractItemView, QApplication, QCheckBox, QHeaderView, QLabel, QMainWindow,
        QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget,
    )
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.figure import Figure
    from mplsoccer import Pitch
    from scipy.ndimage import gaussian_filter

    BG, PANEL, FG, ACCENT, MUTED = "#161b22", "#1e2329", "#e6edf3", "#2f81f7", "#8b98a8"

    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyle("Fusion")
    pal = QPalette()
    for role, col in [
        (QPalette.Window, PANEL), (QPalette.WindowText, FG), (QPalette.Base, BG),
        (QPalette.AlternateBase, "#1c2128"), (QPalette.Text, FG),
        (QPalette.Button, "#2b3138"), (QPalette.ButtonText, FG),
        (QPalette.Highlight, ACCENT), (QPalette.HighlightedText, "#ffffff"),
    ]:
        pal.setColor(role, QColor(col))
    app.setPalette(pal)
    app.setStyleSheet(
        "QTabBar::tab{padding:8px 18px;} "
        "QHeaderView::section{background:#2b3138;color:#e6edf3;padding:6px;border:0;font-weight:600;} "
        "QTableWidget{gridline-color:#2b3138;}"
    )

    common = dict(pitch_color=BG, line_color="#6e7b8b", linewidth=1.5)
    if data.pitch_kind == "statsbomb":
        pitch = Pitch(pitch_type="statsbomb", **common)
        PL, PW = 120, 80
    else:
        pitch = Pitch(pitch_type="custom", pitch_length=105, pitch_width=68, **common)
        PL, PW = 105, 68

    def fresh_axes(fig):
        ax = fig.add_subplot(111)
        pitch.draw(ax=ax)
        return ax

    def title(ax, text):
        ax.set_title(text, color=FG, fontsize=13, pad=12)

    # --- Tab 1: gol subiti ------------------------------------------------
    def conceded_tab() -> QWidget:
        fig = Figure(figsize=(10, 7), facecolor=BG, layout="constrained")
        ax = fresh_axes(fig)
        goals = data.conceded
        title(ax, f"Gol subiti: {len(goals)} (porta difesa a destra)\nDimensione del punto = xG del tiro, autogol esclusi")
        if goals:
            sizes = lambda g: [60 + 700 * (x["xg"] if x["xg"] is not None else 0.1) for x in g]
            op = [g for g in goals if not g["penalty"]]
            pens = [g for g in goals if g["penalty"]]
            if op:
                pitch.scatter([g["x"] for g in op], [g["y"] for g in op], s=sizes(op), c="#ff5c5c",
                              edgecolors="white", linewidth=1, alpha=0.85, ax=ax, label="Azione / palla inattiva")
            if pens:
                pitch.scatter([g["x"] for g in pens], [g["y"] for g in pens], s=sizes(pens), c="#ffd24d",
                              marker="s", edgecolors="white", linewidth=1, alpha=0.9, ax=ax, label="Rigore")
            leg = ax.legend(loc="lower center", facecolor=BG, edgecolor="#6e7b8b", labelcolor=FG,
                            ncol=2, bbox_to_anchor=(0.5, -0.02))
            for h in leg.legend_handles:
                h.set_sizes([60])
        else:
            ax.text(PL / 2, PW / 2, "Nessun gol subito", color=FG, ha="center", va="center", fontsize=14)
        return FigureCanvasQTAgg(fig)

    # --- Tab 2: tabella giocatori ----------------------------------------
    class NumItem(QTableWidgetItem):
        def __init__(self, value, text):
            super().__init__(text)
            self.setData(Qt.ItemDataRole.UserRole, -1.0 if value is None else float(value))
            self.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        def __lt__(self, other):
            return self.data(Qt.ItemDataRole.UserRole) < other.data(Qt.ItemDataRole.UserRole)

    def players_tab() -> QWidget:
        headers = ["Giocatore", "Partite", "Minuti / partita", "Tocchi / partita", "Tocchi / minuto"]
        table = QTableWidget(len(data.players), len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setAlternatingRowColors(True)
        table.verticalHeader().setVisible(False)
        fmt = lambda v, f: "—" if v is None else format(v, f)
        for r, p in enumerate(data.players):
            table.setItem(r, 0, QTableWidgetItem(p["name"]))
            table.setItem(r, 1, NumItem(p["matches"], str(p["matches"])))
            table.setItem(r, 2, NumItem(p["avg_min"], fmt(p["avg_min"], ".1f")))
            table.setItem(r, 3, NumItem(p["avg_touches"], fmt(p["avg_touches"], ".1f")))
            table.setItem(r, 4, NumItem(p["ratio"], fmt(p["ratio"], ".3f")))
        hdr = table.horizontalHeader()
        hdr.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        table.setSortingEnabled(True)
        table.sortItems(2, Qt.SortOrder.DescendingOrder)
        table.verticalHeader().setDefaultSectionSize(30)
        return table

    # --- Tab 3: heatmap ---------------------------------------------------
    def heat_tab() -> QWidget:
        fig = Figure(figsize=(10, 7), facecolor=BG, layout="constrained")
        canvas = FigureCanvasQTAgg(fig)
        box = QWidget()
        lay = QVBoxLayout(box)
        lay.setContentsMargins(0, 0, 0, 0)
        smooth_cb = None
        if data.heat_grid is not None:
            smooth_cb = QCheckBox("Interpolazione (solo estetica: la risoluzione dei dati resta 16x12)")
            smooth_cb.setChecked(True)
            lay.addWidget(smooth_cb)
        lay.addWidget(canvas, 1)

        def render():
            fig.clf()
            ax = fresh_axes(fig)
            title(ax, data.heat_title + " - la squadra attacca verso destra")
            img = None
            if data.heat_grid is not None and data.heat_grid.sum() > 0:
                xl, yl = ax.get_xlim(), ax.get_ylim()
                img = ax.imshow(data.heat_grid.T, origin="lower", extent=(0, PL, 0, PW),
                                cmap="inferno", zorder=0.6, vmin=0, aspect="equal",
                                interpolation="bicubic" if smooth_cb and smooth_cb.isChecked() else "nearest")
                ax.set_xlim(xl); ax.set_ylim(yl)
            elif data.passes_xy is not None and len(data.passes_xy):
                xy = data.passes_xy
                stats = pitch.bin_statistic(xy[:, 0], xy[:, 1], statistic="count", bins=SB_HEAT_BINS)
                stats["statistic"] = gaussian_filter(stats["statistic"], SB_HEAT_SIGMA)
                # vmax al 99.5 percentile: i calci d'angolo non schiacciano la scala
                vmax = float(np.percentile(stats["statistic"], 99.5)) or None
                img = pitch.heatmap(stats, ax=ax, cmap="inferno", edgecolors=None,
                                    vmax=vmax, zorder=0.6)
            else:
                ax.text(PL / 2, PW / 2, "Dati heatmap non disponibili", color=FG, ha="center", va="center")
            if img is not None:
                cb = fig.colorbar(img, ax=ax, shrink=0.7, pad=0.02)
                cb.ax.tick_params(colors=FG)
                cb.outline.set_edgecolor("#6e7b8b")
                cb.set_label("Intensita'", color=FG)
            canvas.draw_idle()

        if smooth_cb:
            smooth_cb.toggled.connect(render)
        render()
        return box

    # --- Finestra ---------------------------------------------------------
    win = QMainWindow()
    win.setWindowTitle(f"Football Viewer - {data.team}")
    win.resize(1200, 820)
    central = QWidget()
    layout = QVBoxLayout(central)
    t = QLabel(data.team)
    f = QFont(); f.setPointSize(20); f.setBold(True)
    t.setFont(f)
    sub = QLabel(f"Stagione {data.season_label}  |  {data.n_matches} partite  |  "
                 f"{len(data.competitions)} competizioni: {', '.join(data.competitions)}")
    sub.setWordWrap(True)
    sub.setStyleSheet(f"color:{MUTED};")
    src = QLabel(f"Fonte: {data.source}" + ("  |  " + " ".join(data.notes) if data.notes else ""))
    src.setWordWrap(True)
    src.setStyleSheet(f"color:{MUTED};font-size:11px;")
    for w in (t, sub, src):
        layout.addWidget(w)
    tabs = QTabWidget()
    tabs.addTab(conceded_tab(), "Gol subiti")
    tabs.addTab(players_tab(), "Giocatori")
    tabs.addTab(heat_tab(), "Heatmap")
    layout.addWidget(tabs, 1)
    win.setCentralWidget(central)
    win.show()
    if os.environ.get("FV_SCREENSHOT"):          # solo per test headless
        for i, name in enumerate(("gol", "giocatori", "heatmap")):
            tabs.setCurrentIndex(i)
            app.processEvents()
            win.grab().save(f"{os.environ['FV_SCREENSHOT']}_{name}.png")
        return
    app.exec()


# ==========================================================================
# CLI
# ==========================================================================
def ask_team(names: dict[str, str]) -> str | None:
    while True:
        q = input("\nNome della squadra (invio per uscire): ").strip()
        if not q:
            return None
        team = resolve_team(q, names)
        if team:
            return team


def main() -> None:
    ap = argparse.ArgumentParser(description="Football Team Viewer")
    ap.add_argument("--source", choices=["pitchapi", "statsbomb"], default="pitchapi",
                    help="pitchapi = stagione corrente (default); statsbomb = dati storici")
    ap.add_argument("--team", help="nome squadra (altrimenti viene chiesto)")
    args = ap.parse_args()
    print("=== Football Team Viewer ===")

    if args.source == "pitchapi":
        client = PitchAPI(get_pitchapi_key())
        try:
            index, cands_by_name = pitchapi_build_index(client)
        except (PitchAPIError, requests.RequestException) as err:
            sys.exit(f"Errore PitchAPI: {err}")
        if not index:
            sys.exit("Nessuna partita trovata per la stagione corrente.")
        names = {norm(n): n for n in cands_by_name}
        team = resolve_team(args.team, names) if args.team else ask_team(names)
        if not team:
            return
        cand = choose_candidate(team, cands_by_name[team])
        label = team if len(cands_by_name[team]) == 1 else f"{team} ({cand['country']})"
        data = collect_pitchapi(client, label, {cand["id"]}, index)
    else:
        try:
            index = sb_build_index()
        except Exception as err:
            sys.exit(f"Impossibile scaricare l'indice StatsBomb: {err}")
        names = {}
        for m in index:
            for n in (m["home_team"]["home_team_name"], m["away_team"]["away_team_name"]):
                names[norm(n)] = n
        team = resolve_team(args.team, names) if args.team else ask_team(names)
        if not team:
            return
        sel = sb_select(index, team)
        if not sel:
            sys.exit(f"Nessuna partita con eventi per {team}.")
        data = collect_statsbomb(team, sel)

    if data.n_matches == 0:
        sys.exit(f"Nessuna partita giocata trovata per {team} nella stagione corrente.")
    print(f"\n{team}: {data.n_matches} partite, {len(data.competitions)} competizioni, "
          f"{len(data.conceded)} gol subiti, {len(data.players)} giocatori")
    run_gui(data)


if __name__ == "__main__":
    main()