"""
rendering.py — Pygame visualisation for LesothoHealthEnv

Draws:
  • A stylised map of Lesotho with 10 district nodes
  • Patient dots on each district (colour = severity)
  • Resource meters (tele-slots, mobile clinics, airlift budget)
  • Live stats HUD: step, reward, fairness score, weather warning
  • Blocked-road overlays when weather events fire
"""

import sys
import math
import numpy as np
from typing import Dict, Any

try:
    import pygame
    import pygame.gfxdraw
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

#  Layout constants
W, H        = 1024, 700
MAP_X       = 40          # map panel left edge
MAP_Y       = 60          # map panel top edge
MAP_W       = 560         # map panel width
MAP_H       = 560         # map panel height
HUD_X       = 630         # right HUD left edge
HUD_W       = 370

#  Color palette 
C_BG        = (15,  20,  30)
C_PANEL     = (22,  30,  45)
C_BORDER    = (40,  55,  80)
C_TEXT      = (210, 220, 235)
C_TEXT_DIM  = (100, 120, 150)
C_ACCENT    = (80,  160, 220)

C_LOW       = (60,  200, 100)   # green  — low severity
C_MEDIUM    = (240, 190, 50)    # yellow — medium
C_CRITICAL  = (230, 60,  60)    # red    — critical
C_RURAL     = (140, 100, 220)   # purple — rural district ring
C_URBAN     = (80,  160, 220)   # blue   — urban district ring
C_BLOCKED   = (200, 80,  40)    # orange — weather-blocked road
C_ROAD      = (50,  65,  90)    # road line

DISTRICT_POSITIONS = [
    (0.45, 0.55),  # Maseru
    (0.35, 0.25),  # Leribe
    (0.50, 0.38),  # Berea
    (0.38, 0.70),  # Mafeteng
    (0.30, 0.82),  # Mohale's Hoek
    (0.38, 0.92),  # Quthing
    (0.62, 0.88),  # Qacha's Nek
    (0.72, 0.30),  # Mokhotlong
    (0.62, 0.50),  # Thaba-Tseka
    (0.45, 0.15),  # Butha-Buthe
]

DISTRICT_NAMES = [
    "Maseru", "Leribe", "Berea", "Mafeteng", "Mohale's Hoek",
    "Quthing", "Qacha's Nek", "Mokhotlong", "Thaba-Tseka", "Butha-Buthe"
]

DISTRICT_RURAL = [False, False, False, True, True, True, True, True, True, True]

# Rough road connections between district indices
ROADS = [
    (0, 1), (0, 2), (0, 3), (1, 2), (1, 9), (2, 8),
    (3, 4), (4, 5), (5, 6), (6, 8), (7, 8), (7, 9),
    (2, 3), (8, 9),
]


def _map_to_px(rx: float, ry: float) -> tuple:
    """Convert relative map position (0-1) to pixel coords."""
    return (
        int(MAP_X + rx * MAP_W),
        int(MAP_Y + ry * MAP_H),
    )


class HealthcareRenderer:
    def __init__(self):
        if not PYGAME_AVAILABLE:
            raise ImportError("pygame is required for rendering. pip install pygame")
        pygame.init()
        pygame.display.set_caption("Lesotho Telemedicine — RL Agent")
        self.screen = pygame.display.set_mode((W, H))
        self.clock  = pygame.time.Clock()
        self.font_lg = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_md = pygame.font.SysFont("monospace", 14)
        self.font_sm = pygame.font.SysFont("monospace", 11)
        self._frame  = None

    def render(self, state: Dict[str, Any]):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        self.screen.fill(C_BG)
        self._draw_title(state)
        self._draw_map(state)
        self._draw_hud(state)
        self._draw_legend()
        pygame.display.flip()
        self.clock.tick(10)

        # Return rgb_array if needed
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2))

    #  Title bar 
    def _draw_title(self, state):
        title = self.font_lg.render(
            "AI Telemedicine Resource Allocation — Lesotho", True, C_ACCENT
        )
        self.screen.blit(title, (MAP_X, 18))
        step_txt = self.font_md.render(
            f"Step {state['step']:>4d} / 200", True, C_TEXT_DIM
        )
        self.screen.blit(step_txt, (MAP_X + MAP_W - 140, 20))

    #  District map 
    def _draw_map(self, state):
        # Map background panel
        pygame.draw.rect(
            self.screen, C_PANEL,
            (MAP_X - 10, MAP_Y - 10, MAP_W + 20, MAP_H + 20), border_radius=12
        )
        pygame.draw.rect(
            self.screen, C_BORDER,
            (MAP_X - 10, MAP_Y - 10, MAP_W + 20, MAP_H + 20), 1, border_radius=12
        )

        blocked = state.get("blocked_districts", set())

        # Draw roads
        for a, b in ROADS:
            pa = _map_to_px(*DISTRICT_POSITIONS[a])
            pb = _map_to_px(*DISTRICT_POSITIONS[b])
            is_blocked = (a in blocked or b in blocked)
            color = C_BLOCKED if is_blocked else C_ROAD
            width = 3 if is_blocked else 1
            pygame.draw.line(self.screen, color, pa, pb, width)

        # Count patients per district and severity
        district_patients = {i: [] for i in range(10)}
        for p in state.get("patient_queue", []):
            district_patients[p["district"]].append(p["severity"])

        # Draw district nodes
        for i, (rx, ry) in enumerate(DISTRICT_POSITIONS):
            cx, cy = _map_to_px(rx, ry)
            is_rural  = DISTRICT_RURAL[i]
            is_blocked = i in blocked

            ring_color = C_RURAL if is_rural else C_URBAN
            radius     = 22

            # Blocked pulse (draw orange ring)
            if is_blocked:
                pygame.draw.circle(self.screen, C_BLOCKED, (cx, cy), radius + 6, 2)

            # Node fill
            pygame.draw.circle(self.screen, C_PANEL, (cx, cy), radius)
            pygame.draw.circle(self.screen, ring_color, (cx, cy), radius, 2)

            # Patient severity dots around the node
            patients = district_patients[i]
            if patients:
                self._draw_patient_dots(cx, cy, patients, radius)

            # District label
            name_short = DISTRICT_NAMES[i].split("'")[0][:8]
            lbl = self.font_sm.render(name_short, True, C_TEXT_DIM)
            lw  = lbl.get_width()
            self.screen.blit(lbl, (cx - lw // 2, cy + radius + 3))

            # Patient count badge
            n = len(patients)
            if n > 0:
                badge_color = (
                    C_CRITICAL if any(s == 2 for s in patients)
                    else C_MEDIUM if any(s == 1 for s in patients)
                    else C_LOW
                )
                pygame.draw.circle(self.screen, badge_color, (cx + radius - 4, cy - radius + 4), 9)
                ntxt = self.font_sm.render(str(n), True, C_BG)
                self.screen.blit(ntxt, (cx + radius - 4 - ntxt.get_width() // 2,
                                        cy - radius + 4 - ntxt.get_height() // 2))

            # Mobile clinic indicator
            if any(d == i for _, d in state.get("mobile_busy", [])):
                pygame.draw.circle(self.screen, C_ACCENT, (cx - radius + 4, cy - radius + 4), 6)
                t = self.font_sm.render("M", True, C_BG)
                self.screen.blit(t, (cx - radius + 4 - t.get_width() // 2,
                                     cy - radius + 4 - t.get_height() // 2))

        # Weather warning overlay
        if state.get("weather", 0) > 0:
            n_blocked = len(blocked)
            warn = self.font_md.render(
                f"⚠ Weather: {n_blocked} district(s) blocked", True, C_BLOCKED
            )
            self.screen.blit(warn, (MAP_X + 4, MAP_Y + MAP_H - 24))

    def _draw_patient_dots(self, cx, cy, severities, radius):
        """Draw small coloured dots orbiting the district node."""
        n = len(severities)
        for k, sev in enumerate(severities[:8]):  # cap at 8 visible
            angle = (2 * math.pi * k / max(n, 1)) - math.pi / 2
            dx = int((radius + 12) * math.cos(angle))
            dy = int((radius + 12) * math.sin(angle))
            color = [C_LOW, C_MEDIUM, C_CRITICAL][sev]
            pygame.draw.circle(self.screen, color, (cx + dx, cy + dy), 5)

    #  HUD panel 
    def _draw_hud(self, state):
        # Panel background
        pygame.draw.rect(
            self.screen, C_PANEL,
            (HUD_X, 60, HUD_W - 10, H - 80), border_radius=10
        )
        pygame.draw.rect(
            self.screen, C_BORDER,
            (HUD_X, 60, HUD_W - 10, H - 80), 1, border_radius=10
        )

        y = 80
        def hdr(text):
            nonlocal y
            t = self.font_lg.render(text, True, C_ACCENT)
            self.screen.blit(t, (HUD_X + 12, y))
            y += 26

        def row(label, value, color=C_TEXT):
            nonlocal y
            lt = self.font_md.render(label, True, C_TEXT_DIM)
            vt = self.font_md.render(str(value), True, color)
            self.screen.blit(lt, (HUD_X + 12, y))
            self.screen.blit(vt, (HUD_X + HUD_W - 20 - vt.get_width(), y))
            y += 22

        def bar(label, value, max_val, color):
            nonlocal y
            lt = self.font_sm.render(label, True, C_TEXT_DIM)
            self.screen.blit(lt, (HUD_X + 12, y))
            bx, by, bw, bh = HUD_X + 160, y + 2, 170, 14
            pygame.draw.rect(self.screen, C_BORDER, (bx, by, bw, bh), border_radius=4)
            fill = int(bw * (value / max(max_val, 1)))
            if fill > 0:
                pygame.draw.rect(self.screen, color, (bx, by, fill, bh), border_radius=4)
            vt = self.font_sm.render(f"{value}/{max_val}", True, C_TEXT_DIM)
            self.screen.blit(vt, (bx + bw + 6, y))
            y += 24

        def sep():
            nonlocal y
            pygame.draw.line(self.screen, C_BORDER,
                             (HUD_X + 10, y + 4), (HUD_X + HUD_W - 20, y + 4))
            y += 14

        #  Live stats 
        hdr("LIVE STATS")
        row("Step",          f"{state['step']} / 200")
        row("Queue size",    len(state.get("patient_queue", [])))
        row("Reward (ep.)",  f"{state.get('episode_reward', 0):.1f}",
            C_LOW if state.get("episode_reward", 0) >= 0 else C_CRITICAL)
        crit = state.get("untreated_critical", 0)
        row("Critical unc.", crit,
            C_CRITICAL if crit >= 4 else C_MEDIUM if crit >= 2 else C_LOW)

        sep()

        #  Resources 
        hdr("RESOURCES")
        bar("Mobile clinics", state.get("mobile_left", 2),    2, C_ACCENT)
        bar("Tele-slots",     state.get("tele_slots", 4),      4, C_RURAL)
        bar("Airlift budget", state.get("airlift_budget", 3),  3, C_CRITICAL)
        y += 4

        sep()

        #  Fairness 
        hdr("FAIRNESS")
        total  = state.get("treated_total",  0)
        urban  = state.get("treated_urban",  0)
        rural  = total - urban
        row("Treated total",  total)
        row("  — urban",      urban,  C_URBAN)
        row("  — rural",      rural,  C_RURAL)

        if total > 0:
            ratio = urban / total
            bar("Urban ratio", int(ratio * 100), 100,
                C_CRITICAL if ratio > 0.70 else C_MEDIUM if ratio > 0.50 else C_LOW)
            # Ideal: ~30% urban (matches population distribution)
            ideal_delta = abs(ratio - 0.30)
            fscore = max(0.0, 1.0 - ideal_delta * 2)
            fcol = C_LOW if fscore > 0.7 else C_MEDIUM if fscore > 0.4 else C_CRITICAL
            row("Fairness score", f"{fscore:.2f}", fcol)

        sep()

        #  Weather 
        hdr("ENVIRONMENT")
        w = state.get("weather", 0)
        row("Weather severity", f"{w:.0%}",
            C_CRITICAL if w > 0.3 else C_MEDIUM if w > 0 else C_LOW)
        blocked = state.get("blocked_districts", set())
        if blocked:
            from environment.custom_env import DISTRICTS
            names = ", ".join(DISTRICTS[b][0] for b in blocked)
            # Word-wrap across two lines
            for chunk in [names[:30], names[30:] if len(names) > 30 else ""]:
                if chunk:
                    t = self.font_sm.render(chunk, True, C_BLOCKED)
                    self.screen.blit(t, (HUD_X + 12, y))
                    y += 18
        else:
            row("Roads", "All clear", C_LOW)

    #  Legend  
    def _draw_legend(self):
        items = [
            (C_LOW,      "Low severity"),
            (C_MEDIUM,   "Medium severity"),
            (C_CRITICAL, "Critical"),
            (C_URBAN,    "Urban district"),
            (C_RURAL,    "Rural district"),
            (C_BLOCKED,  "Blocked road"),
            (C_ACCENT,   "Mobile clinic en-route"),
        ]
        x, y = MAP_X, MAP_Y + MAP_H + 18
        for color, label in items:
            pygame.draw.circle(self.screen, color, (x + 6, y + 7), 6)
            t = self.font_sm.render(label, True, C_TEXT_DIM)
            self.screen.blit(t, (x + 16, y))
            x += t.get_width() + 30
            if x > MAP_X + MAP_W - 60:
                x  = MAP_X
                y += 20

    def close(self):
        pygame.quit()
