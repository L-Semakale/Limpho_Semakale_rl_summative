import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Tuple, Dict, Any


# District definitions: (name, is_rural, base_connectivity, x_pos, y_pos)
DISTRICTS = [
    ("Maseru",        False, 0.9, 0.45, 0.55),
    ("Leribe",        False, 0.8, 0.35, 0.25),
    ("Berea",         False, 0.75, 0.50, 0.38),
    ("Mafeteng",      True,  0.5, 0.38, 0.70),
    ("Mohale's Hoek", True,  0.4, 0.30, 0.82),
    ("Quthing",       True,  0.3, 0.38, 0.92),
    ("Qacha's Nek",   True,  0.2, 0.62, 0.88),
    ("Mokhotlong",    True,  0.15, 0.72, 0.30),
    ("Thaba-Tseka",   True,  0.2, 0.62, 0.50),
    ("Butha-Buthe",   True,  0.3, 0.45, 0.15),
]

N_DISTRICTS      = len(DISTRICTS)
MAX_STEPS        = 200
MAX_QUEUE        = 20
MAX_UNTREATED    = 8
MOBILE_CLINICS   = 2
TELE_SLOTS       = 4
AIRLIFT_BUDGET   = 3

# Actions
ACTION_TELECONSULT = 0
ACTION_MOBILE      = 1
ACTION_SCHEDULE    = 2
ACTION_IGNORE      = 3
ACTION_AIRLIFT     = 4
N_ACTIONS          = 5


class LesothoHealthEnv(gym.Env):
    """
    Telemedicine Resource Allocation Environment — Lesotho

    The agent manages scarce healthcare resources across 10 districts.
    Each step it observes the most urgent patient and decides how to respond.
    Patients deteriorate if untreated; resources are limited and replenish
    slowly.  Stochastic weather events block mountain roads and cut connectivity.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()
        self.render_mode = render_mode
        self._renderer = None

        #  Observation space (12 features, all normalised to [0,1]) 
        # 0  patient_severity        0=low, 0.5=medium, 1=critical
        # 1  patient_location        0=urban, 1=rural
        # 2  wait_time               steps waiting / MAX_STEPS
        # 3  connectivity            0-1
        # 4  queue_size              n / MAX_QUEUE
        # 5  mobile_clinics_left     n / MOBILE_CLINICS
        # 6  tele_slots_left         n / TELE_SLOTS
        # 7  weather_penalty         0=clear, 1=severe blockage
        # 8  district_id             district / N_DISTRICTS
        # 9  urban_ratio_bias        fraction of treated so far that were urban
        # 10 patient_age_group       0=child, 0.5=adult, 1=elderly
        # 11 time_of_day             step % 24 / 24
        low  = np.zeros(12, dtype=np.float32)
        high = np.ones(12,  dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        #  Action space 
        self.action_space = spaces.Discrete(N_ACTIONS)

        # Internal state (initialised in reset)
        self._patient_queue: list  = []
        self._mobile_busy: list    = []   # [(steps_remaining, district_id)]
        self._step: int            = 0
        self._tele_slots: int      = TELE_SLOTS
        self._mobile_left: int     = MOBILE_CLINICS
        self._airlift_budget: int  = AIRLIFT_BUDGET
        self._weather: float       = 0.0
        self._blocked_districts: set = set()
        self._treated_urban: int   = 0
        self._treated_total: int   = 0
        self._untreated_critical: int = 0
        self._episode_reward: float  = 0.0
        self._fairness_log: list   = []

    
    # Gym API
    

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step               = 0
        self._tele_slots         = TELE_SLOTS
        self._mobile_left        = MOBILE_CLINICS
        self._airlift_budget     = AIRLIFT_BUDGET
        self._weather            = 0.0
        self._blocked_districts  = set()
        self._treated_urban      = 0
        self._treated_total      = 0
        self._untreated_critical = 0
        self._episode_reward     = 0.0
        self._mobile_busy        = []
        self._fairness_log       = []

        # Seed the initial queue with 5 patients
        self._patient_queue = []
        for _ in range(5):
            self._patient_queue.append(self._spawn_patient())

        obs  = self._get_obs()
        info = self._get_info()
        if self.render_mode == "human":
            self._render_frame()
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        self._step += 1

        #  Resolve mobile clinics in transit 
        still_busy = []
        for steps_left, dist in self._mobile_busy:
            if steps_left <= 1:
                self._mobile_left = min(MOBILE_CLINICS, self._mobile_left + 1)
            else:
                still_busy.append((steps_left - 1, dist))
        self._mobile_busy = still_busy

        #  Replenish tele-slots every 4 steps 
        if self._step % 4 == 0:
            self._tele_slots = min(TELE_SLOTS, self._tele_slots + 1)

        #  Stochastic weather event every 10 steps 
        if self._step % 10 == 0:
            self._update_weather()

        #  Spawn new patients (Poisson-ish) ---
        n_new = self.np_random.poisson(1.5)
        for _ in range(n_new):
            if len(self._patient_queue) < MAX_QUEUE:
                self._patient_queue.append(self._spawn_patient())

        #  Deteriorate untreated patients 
        for p in self._patient_queue:
            p["wait"] += 1
            # Medium → critical after 5 ignored steps
            if p["severity"] == 1 and p["wait"] >= 5:
                p["severity"] = 2
            # Low → medium after 8 ignored steps
            if p["severity"] == 0 and p["wait"] >= 8:
                p["severity"] = 1

        #  Get current patient (most urgent) ---
        if not self._patient_queue:
            # No patients: small idle penalty
            reward = -1.0
            obs = self._get_obs()
            terminated = self._is_terminal()
            if self.render_mode == "human":
                self._render_frame()
            return obs, reward, terminated, False, self._get_info()

        patient = self._get_priority_patient()
        reward  = self._apply_action(action, patient)

        self._episode_reward += reward

        # Track fairness
        if self._treated_total > 0:
            self._fairness_log.append(self._treated_urban / self._treated_total)

        # Count uncritical patients still waiting
        self._untreated_critical = sum(
            1 for p in self._patient_queue if p["severity"] == 2
        )

        terminated = self._is_terminal()
        truncated  = self._step >= MAX_STEPS

        obs  = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    
    # Internal helpers
    

    def _spawn_patient(self) -> Dict:
        dist_id = int(self.np_random.integers(0, N_DISTRICTS))
        dist    = DISTRICTS[dist_id]
        # Rural districts more likely to be severe (access deprivation)
        if dist[1]:  # is_rural
            severity = int(self.np_random.choice([0, 1, 2], p=[0.2, 0.4, 0.4]))
        else:
            severity = int(self.np_random.choice([0, 1, 2], p=[0.4, 0.4, 0.2]))

        connectivity = float(dist[2]) * float(
            self.np_random.uniform(0.7, 1.0)
        )
        # Weather degrades connectivity for mountain districts
        if dist_id in self._blocked_districts:
            connectivity *= 0.3

        return {
            "severity":     severity,       # 0=low, 1=medium, 2=critical
            "district":     dist_id,
            "is_rural":     dist[1],
            "wait":         0,
            "connectivity": connectivity,
            "age_group":    int(self.np_random.choice([0, 1, 2], p=[0.2, 0.6, 0.2])),
        }

    def _get_priority_patient(self) -> Dict:
        """Return most urgent patient (severity desc, then wait desc)."""
        return max(
            self._patient_queue,
            key=lambda p: (p["severity"] * 1000 + p["wait"])
        )

    def _apply_action(self, action: int, patient: Dict) -> float:
        severity     = patient["severity"]
        is_rural     = patient["is_rural"]
        connectivity = patient["connectivity"]
        dist_id      = patient["district"]
        reward       = 0.0

        if action == ACTION_TELECONSULT:
            if self._tele_slots > 0 and connectivity > 0.4:
                self._tele_slots -= 1
                self._patient_queue.remove(patient)
                reward += 10.0
                reward += 10.0 * (severity / 2.0)   # bonus for severity
                reward += 8.0  if is_rural else 0.0  # equity bonus
                self._record_treated(patient)
            else:
                # Failed teleconsult: wasted slot or no connectivity
                reward -= 5.0 if connectivity <= 0.4 else -3.0

        elif action == ACTION_MOBILE:
            if (self._mobile_left > 0
                    and dist_id not in self._blocked_districts):
                self._mobile_left -= 1
                travel = 3 if is_rural else 1
                self._mobile_busy.append((travel, dist_id))
                self._patient_queue.remove(patient)
                reward += 15.0 if is_rural else 8.0
                reward += 10.0 * (severity / 2.0)
                self._record_treated(patient)
            elif dist_id in self._blocked_districts:
                reward -= 4.0  # tried but road blocked
            else:
                reward -= 3.0  # no mobile clinics left

        elif action == ACTION_SCHEDULE:
            if severity == 0:
                reward += 3.0  # fine to defer low-severity
            elif severity == 1:
                reward -= 5.0  # medium should not be deferred lightly
            else:
                reward -= 20.0  # critical — never just schedule

        elif action == ACTION_IGNORE:
            if severity == 0:
                reward -= 1.0
            elif severity == 1:
                reward -= 8.0
            else:
                reward -= 20.0  # critical ignore: heavy penalty

        elif action == ACTION_AIRLIFT:
            if self._airlift_budget > 0 and severity == 2:
                self._airlift_budget -= 1
                self._patient_queue.remove(patient)
                reward += 25.0 + (10.0 if is_rural else 0.0)
                self._record_treated(patient)
            elif severity < 2:
                reward -= 10.0  # wasteful airlift
            else:
                reward -= 5.0   # budget exhausted

        # Equity penalty: if agent is systematically biasing toward urban
        if self._treated_total > 5:
            urban_ratio = self._treated_urban / self._treated_total
            if urban_ratio > 0.7:
                reward -= 15.0  # strong bias penalty

        return float(reward)

    def _record_treated(self, patient: Dict):
        self._treated_total += 1
        if not patient["is_rural"]:
            self._treated_urban += 1

    def _update_weather(self):
        """Randomly block 0-2 mountain/rural districts."""
        self._blocked_districts = set()
        self._weather = 0.0
        if self.np_random.random() < 0.4:
            n_blocked = int(self.np_random.integers(1, 3))
            rural_ids = [i for i, d in enumerate(DISTRICTS) if d[1]]
            blocked   = self.np_random.choice(rural_ids, size=min(n_blocked, len(rural_ids)), replace=False)
            self._blocked_districts = set(int(b) for b in blocked)
            self._weather = len(self._blocked_districts) / N_DISTRICTS

    def _is_terminal(self) -> bool:
        return self._untreated_critical >= MAX_UNTREATED

    def _get_obs(self) -> np.ndarray:
        if self._patient_queue:
            p = self._get_priority_patient()
            severity     = p["severity"] / 2.0
            location     = float(p["is_rural"])
            wait         = min(p["wait"] / MAX_STEPS, 1.0)
            connectivity = p["connectivity"]
            dist_id      = p["district"] / N_DISTRICTS
            age_group    = p["age_group"] / 2.0
        else:
            severity = location = wait = connectivity = dist_id = age_group = 0.0

        urban_bias = (
            self._treated_urban / self._treated_total
            if self._treated_total > 0 else 0.5
        )

        obs = np.array([
            severity,
            location,
            wait,
            connectivity,
            min(len(self._patient_queue) / MAX_QUEUE, 1.0),
            self._mobile_left  / MOBILE_CLINICS,
            self._tele_slots   / TELE_SLOTS,
            self._weather,
            dist_id,
            urban_bias,
            age_group,
            (self._step % 24) / 24.0,
        ], dtype=np.float32)

        return obs

    def _get_info(self) -> Dict:
        return {
            "step":               self._step,
            "queue_size":         len(self._patient_queue),
            "mobile_left":        self._mobile_left,
            "tele_slots":         self._tele_slots,
            "airlift_budget":     self._airlift_budget,
            "untreated_critical": self._untreated_critical,
            "treated_total":      self._treated_total,
            "treated_urban":      self._treated_urban,
            "weather":            self._weather,
            "blocked_districts":  list(self._blocked_districts),
            "episode_reward":     self._episode_reward,
            "fairness_score": (
                1.0 - abs(
                    self._treated_urban / self._treated_total - 0.3
                )
                if self._treated_total > 0 else 1.0
            ),
        }

    def _render_frame(self):
        from environment.rendering import HealthcareRenderer
        if self._renderer is None:
            self._renderer = HealthcareRenderer()
        state = {
            "patient_queue":      self._patient_queue,
            "blocked_districts":  self._blocked_districts,
            "mobile_busy":        self._mobile_busy,
            "mobile_left":        self._mobile_left,
            "tele_slots":         self._tele_slots,
            "airlift_budget":     self._airlift_budget,
            "weather":            self._weather,
            "step":               self._step,
            "episode_reward":     self._episode_reward,
            "treated_total":      self._treated_total,
            "treated_urban":      self._treated_urban,
            "untreated_critical": self._untreated_critical,
        }
        return self._renderer.render(state)
