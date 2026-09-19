"""
gesture_hud.py
====================================================
Gesture-Controlled Heads-Up Display (HUD) with
Time-Series Telemetry Logging.

Pipeline:
    Layer 1: Webcam capture (OpenCV)
    Layer 2: Hand landmark extraction (MediaPipe)
    Layer 3: Euclidean gesture classification + 3-frame debounce state machine
    Layer 4: Alpha-blended HUD overlay + SQLite telemetry logger

Run:
    python gesture_hud.py
    
benchmark output:
    python analyze_telemetry.py

Controls:
    q  -> quit
    b  -> mark start/stop of a benchmark run (writes a row to benchmark_summary)
    l  -> cycle the lighting_condition label attached to the current benchmark run
    x  -> toggle real OS-level system control on/off (starts OFF for safety)
    p  -> toggle the side-panel dashboard on/off

Gestures:
    pinch        -> SELECT      (Enter key)
    open_palm    -> PLAY/PAUSE
    point        -> MUTE        (index finger only, other 3 fingers curled)
    peace        -> SNAPSHOT    (saves current frame to snapshots/, works even with system control off)
    thumbs_up    -> VOLUME UP
    thumbs_down  -> VOLUME DOWN
    swipe_left   -> PREV TRACK  (fast horizontal hand movement)
    swipe_right  -> NEXT TRACK
    fist         -> recognized but unmapped (no command fires)
====================================================
"""

import cv2
import numpy as np
import mediapipe as mp
import sqlite3
import time
import math
import os
from datetime import datetime
from collections import deque

try:
    import pyautogui
    pyautogui.FAILSAFE = False
    PYAUTOGUI_AVAILABLE = True
except ImportError:
    PYAUTOGUI_AVAILABLE = False

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
DB_PATH = "telemetry.db"
CAM_INDEX = 0
FRAME_W, FRAME_H = 640, 480     # lower res = lower latency; bump to 1280x720 only if your machine can keep up

DRAW_LANDMARKS = True           # set False for latency testing/benchmarking (skeleton drawing adds overhead)
DEBUG_GESTURE = True            # shows live fingers-extended count + pinch ratio on screen; turn off once tuned
SHOW_SIDE_PANEL = True          # dedicated dashboard panel rendered next to the camera feed

PANEL_WIDTH = 280               # width in px of the side-panel dashboard

SYSTEM_CONTROL_ENABLED = False  # start OFF for safety; toggle live with 'x'. When True, gestures send real
                                 # OS media/volume keys via pyautogui, not just on-screen labels.

PINCH_THRESHOLD_RATIO = 0.40    # thumb-tip<->index-tip distance, normalized by palm size (scale-invariant)
THUMB_VERTICAL_RATIO = 0.35     # thumb-tip vs thumb-mcp vertical offset, normalized by palm size,
                                 # used to tell thumbs_up/thumbs_down apart from a plain fist

SWIPE_WINDOW_SEC = 0.45         # time window over which we look for horizontal wrist movement
SWIPE_DISTANCE_RATIO = 0.22     # min wrist displacement (fraction of frame width) within the window to count as a swipe
SWIPE_COOLDOWN_SEC = 1.0        # minimum time between two swipe triggers

DEBOUNCE_FRAMES = 3             # consecutive frames required to confirm a gesture
LATENCY_LOG_BUFFER = 25         # rows buffered before a single SQLite commit

LIGHTING_CONDITIONS = ["normal_indoor", "dim_room", "bright_sunlight"]

# MediaPipe landmark indices we care about
WRIST = 0
THUMB_MCP = 2
THUMB_TIP = 4
INDEX_PIP = 6
INDEX_TIP = 8
MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_TIP = 12
RING_PIP = 14
RING_TIP = 16
PINKY_PIP = 18
PINKY_TIP = 20
# (tip, pip) pairs for the four non-thumb fingers used in extension checks
FINGER_JOINTS = [(INDEX_TIP, INDEX_PIP), (MIDDLE_TIP, MIDDLE_PIP),
                  (RING_TIP, RING_PIP), (PINKY_TIP, PINKY_PIP)]


# ----------------------------------------------------------------------
# LAYER 4a: DATABASE
# ----------------------------------------------------------------------
class TelemetryDB:
    """Owns the SQLite connection and both tables from the schema."""

    def __init__(self, db_path=DB_PATH):
        self.conn = sqlite3.connect(db_path)
        self.cur = self.conn.cursor()
        self._create_schema()
        self._buffer = []

    def _create_schema(self):
        self.cur.execute("""
            CREATE TABLE IF NOT EXISTS telemetry_logs (
                log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                capture_latency_ms REAL NOT NULL,
                inference_latency_ms REAL NOT NULL,
                total_latency_ms REAL NOT NULL,
                detected_gesture TEXT
            )
        """)
        self.cur.execute("""
            CREATE TABLE IF NOT EXISTS benchmark_summary (
                test_id INTEGER PRIMARY KEY AUTOINCREMENT,
                lighting_condition TEXT NOT NULL,
                total_frames_tested INTEGER NOT NULL,
                false_positives INTEGER NOT NULL,
                avg_latency_ms REAL NOT NULL,
                run_started_at TEXT,
                run_ended_at TEXT
            )
        """)
        self.conn.commit()

    def log_frame(self, capture_ms, inference_ms, total_ms, gesture):
        self._buffer.append((
            datetime.now().isoformat(timespec="milliseconds"),
            capture_ms, inference_ms, total_ms, gesture or "none"
        ))
        if len(self._buffer) >= LATENCY_LOG_BUFFER:
            self.flush()

    def flush(self):
        if not self._buffer:
            return
        self.cur.executemany("""
            INSERT INTO telemetry_logs
                (timestamp, capture_latency_ms, inference_latency_ms, total_latency_ms, detected_gesture)
            VALUES (?, ?, ?, ?, ?)
        """, self._buffer)
        self.conn.commit()
        self._buffer.clear()

    def write_benchmark_summary(self, lighting_condition, total_frames, false_positives,
                                 avg_latency_ms, started_at, ended_at):
        self.cur.execute("""
            INSERT INTO benchmark_summary
                (lighting_condition, total_frames_tested, false_positives, avg_latency_ms,
                 run_started_at, run_ended_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (lighting_condition, total_frames, false_positives, avg_latency_ms, started_at, ended_at))
        self.conn.commit()

    def close(self):
        self.flush()
        self.conn.close()


# ----------------------------------------------------------------------
# LAYER 3: GEOMETRY + GESTURE CLASSIFICATION
# ----------------------------------------------------------------------
def euclidean_distance(p1, p2):
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2 + (p1.z - p2.z) ** 2)


def is_finger_extended(landmarks, tip_idx, pip_idx, wrist_idx=WRIST):
    """A finger counts as 'extended' if its tip is farther from the wrist
    than its own PIP knuckle is. This is scale-invariant: it works the same
    whether the hand is close to or far from the camera, and for any hand size,
    unlike a fixed absolute-distance threshold."""
    wrist = landmarks[wrist_idx]
    return euclidean_distance(landmarks[tip_idx], wrist) > euclidean_distance(landmarks[pip_idx], wrist)


def classify_gesture(landmarks, debug=False):
    """
    landmarks: mediapipe NormalizedLandmarkList.landmark (21 points)
    Returns one of: 'pinch', 'open_palm', 'fist', 'peace', 'point', 'thumbs_up', 'thumbs_down', None
    (Swipe gestures are motion-based and handled separately by SwipeDetector, not here.)
    """
    wrist = landmarks[WRIST]

    # Palm-size reference (wrist <-> middle-finger MCP) used to normalize distances
    # so thresholds don't depend on hand size / distance from camera.
    palm_size = euclidean_distance(landmarks[MIDDLE_MCP], wrist)
    if palm_size < 1e-6:
        return (None, None) if debug else None

    # Only the four non-thumb fingers decide fist / open_palm / peace / point. The thumb
    # is deliberately excluded from THIS check: its joint folds sideways across the palm
    # rather than straight back toward the wrist, so a wrist-distance check
    # unreliably reads it as "extended" even inside a tight fist.
    index_ext, middle_ext, ring_ext, pinky_ext = [
        is_finger_extended(landmarks, tip, pip) for tip, pip in FINGER_JOINTS
    ]
    curled_count = sum([index_ext, middle_ext, ring_ext, pinky_ext])  # 0..4 extended

    pinch_dist_ratio = euclidean_distance(landmarks[THUMB_TIP], landmarks[INDEX_TIP]) / palm_size

    # Thumb vertical offset (image y grows downward), normalized by palm size.
    # Only meaningful for disambiguating thumbs_up/down from a plain fist.
    thumb_dy_ratio = (landmarks[THUMB_TIP].y - landmarks[THUMB_MCP].y) / palm_size

    if curled_count == 0:
        if thumb_dy_ratio < -THUMB_VERTICAL_RATIO:
            gesture = "thumbs_up"
        elif thumb_dy_ratio > THUMB_VERTICAL_RATIO:
            gesture = "thumbs_down"
        else:
            gesture = "fist"
    elif index_ext and middle_ext and not ring_ext and not pinky_ext:
        gesture = "peace"
    elif index_ext and not middle_ext and not ring_ext and not pinky_ext:
        gesture = "point"
    elif pinch_dist_ratio < PINCH_THRESHOLD_RATIO:
        gesture = "pinch"
    elif curled_count >= 4:
        gesture = "open_palm"
    else:
        gesture = None

    if debug:
        return gesture, {"fingers_extended": curled_count, "pinch_ratio": round(pinch_dist_ratio, 3),
                          "thumb_dy_ratio": round(thumb_dy_ratio, 3)}
    return gesture


# ----------------------------------------------------------------------
# LAYER 3b: MOTION-BASED SWIPE DETECTION
# ----------------------------------------------------------------------
class SwipeDetector:
    """
    Tracks wrist x-position over a short rolling time window and fires
    'swipe_left' / 'swipe_right' when the hand moves far enough, fast enough.
    This is separate from classify_gesture() because a swipe is a MOTION
    pattern across frames, not a single frame's hand shape.
    """

    def __init__(self, window_sec=SWIPE_WINDOW_SEC, distance_ratio=SWIPE_DISTANCE_RATIO,
                 cooldown_sec=SWIPE_COOLDOWN_SEC):
        self.window_sec = window_sec
        self.distance_ratio = distance_ratio
        self.cooldown_sec = cooldown_sec
        self.history = deque()   # (timestamp, wrist_x)
        self.last_fire_time = 0.0

    def update(self, wrist_x_norm, hand_present):
        now = time.perf_counter()

        if not hand_present:
            self.history.clear()
            return None

        self.history.append((now, wrist_x_norm))
        while self.history and now - self.history[0][0] > self.window_sec:
            self.history.popleft()

        if now - self.last_fire_time < self.cooldown_sec or len(self.history) < 2:
            return None

        oldest_t, oldest_x = self.history[0]
        dx = wrist_x_norm - oldest_x

        if abs(dx) < self.distance_ratio:
            return None

        self.last_fire_time = now
        self.history.clear()
        return "swipe_right" if dx > 0 else "swipe_left"


# ----------------------------------------------------------------------
# LAYER 3: TEMPORAL DEBOUNCE STATE MACHINE
# ----------------------------------------------------------------------
class DebounceStateMachine:
    """
    Requires a gesture to persist for DEBOUNCE_FRAMES consecutive frames
    before it is considered "confirmed" / command-triggering.
    Guards against the Midas-touch false-positive problem.
    """

    def __init__(self, required_frames=DEBOUNCE_FRAMES):
        self.required_frames = required_frames
        self.candidate = None
        self.streak = 0
        self.confirmed = None
        self.last_triggered = None  # gesture that most recently fired a command

    def update(self, raw_gesture):
        if raw_gesture == self.candidate:
            self.streak += 1
        else:
            self.candidate = raw_gesture
            self.streak = 1

        just_triggered = None
        if self.streak >= self.required_frames and raw_gesture is not None:
            self.confirmed = raw_gesture
            if self.confirmed != self.last_triggered:
                just_triggered = self.confirmed  # rising edge -> fire command once
                self.last_triggered = self.confirmed
        elif raw_gesture is None and self.streak >= self.required_frames:
            self.confirmed = None
            self.last_triggered = None

        return self.confirmed, just_triggered


# ----------------------------------------------------------------------
# LAYER 4b: HUD RENDERING
# ----------------------------------------------------------------------
COMMAND_MAP = {
    "pinch": "SELECT",
    "open_palm": "PLAY / PAUSE",
    "point": "MUTE",
    "peace": "SNAPSHOT",
    "thumbs_up": "VOLUME UP",
    "thumbs_down": "VOLUME DOWN",
    "swipe_left": "PREV TRACK",
    "swipe_right": "NEXT TRACK",
}

# Maps gesture -> pyautogui key name (or None for actions handled specially, like snapshot)
SYSTEM_KEY_MAP = {
    "pinch": "enter",
    "open_palm": "playpause",
    "point": "volumemute",
    "thumbs_up": "volumeup",
    "thumbs_down": "volumedown",
    "swipe_left": "prevtrack",
    "swipe_right": "nexttrack",
}


def execute_system_action(gesture, frame):
    """
    Fires a REAL OS-level action for the given gesture. Only called on the
    rising edge of a confirmed gesture (once per gesture activation), and only
    when SYSTEM_CONTROL_ENABLED is True. Returns a short status string for logging.
    """
    if gesture == "peace":
        os.makedirs("snapshots", exist_ok=True)
        path = os.path.join("snapshots", f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        cv2.imwrite(path, frame)
        return f"saved {path}"

    key = SYSTEM_KEY_MAP.get(gesture)
    if key is None:
        return None
    if not PYAUTOGUI_AVAILABLE:
        return "pyautogui not installed"
    try:
        pyautogui.press(key)
        return f"pressed '{key}'"
    except Exception as e:
        return f"error: {e}"

ICON_COLOR = (0, 255, 180)      # BGR - cyan/green HUD accent
TEXT_COLOR = (255, 255, 255)
PANEL_COLOR = (20, 20, 20)


def draw_hud(frame, confirmed_gesture, just_triggered, fps, total_latency_ms, lighting_label, debug_info=None):
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Semi-transparent top status bar
    cv2.rectangle(overlay, (0, 0), (w, 115), PANEL_COLOR, -1)
    frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)

    cv2.putText(frame, "GESTURE HUD", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, ICON_COLOR, 2)
    skeleton_state = "ON" if DRAW_LANDMARKS else "OFF"
    sys_state = "LIVE" if SYSTEM_CONTROL_ENABLED else "OFF (press x)"
    sys_color = (0, 0, 255) if SYSTEM_CONTROL_ENABLED else TEXT_COLOR
    cv2.putText(frame, f"FPS: {fps:5.1f}   Latency: {total_latency_ms:5.1f} ms   "
                        f"Lighting: {lighting_label}   Skeleton: {skeleton_state}",
                (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, TEXT_COLOR, 1)
    cv2.putText(frame, f"System control: {sys_state}", (w - 260, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, sys_color, 2)

    if debug_info is not None:
        cv2.putText(frame, f"[debug] fingers: {debug_info['fingers_extended']}/4  "
                            f"pinch: {debug_info['pinch_ratio']} (<{PINCH_THRESHOLD_RATIO})  "
                            f"thumb_dy: {debug_info['thumb_dy_ratio']} (up<-{THUMB_VERTICAL_RATIO}/down>{THUMB_VERTICAL_RATIO})",
                    (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)

    # Current confirmed gesture / command panel (bottom-left)
    label = confirmed_gesture.upper() if confirmed_gesture else "..."
    command = COMMAND_MAP.get(confirmed_gesture, "-")
    panel_y = h - 70
    cv2.rectangle(frame, (0, panel_y), (400, h), PANEL_COLOR, -1)
    cv2.putText(frame, f"Gesture: {label}", (15, panel_y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, ICON_COLOR, 2)
    cv2.putText(frame, f"Command: {command}", (15, panel_y + 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, TEXT_COLOR, 1)

    # Flash a confirmation ring when a command actually fires (rising edge)
    if just_triggered:
        cv2.circle(frame, (w - 70, 70), 40, ICON_COLOR, 4)
        cv2.putText(frame, "OK", (w - 90, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.8, ICON_COLOR, 2)

    return frame


def _icon_color(active):
    """Bright accent when a status is active/flashing, dim grey when idle."""
    return (0, 255, 255) if active else (95, 95, 95)


def build_side_panel(width, height, state, now):
    """
    Renders a dedicated automotive-style dashboard (speedometer/select ring,
    play-pause, volume bars, nav chevrons, snapshot indicator) as its own
    image, meant to be placed next to the camera feed rather than overlaid
    on top of it. `state` is the hud_state dict maintained in main().
    """
    panel = np.full((height, width, 3), PANEL_COLOR, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(panel, "VEHICLE HUD", (20, 40), font, 0.8, ICON_COLOR, 2)
    cv2.line(panel, (15, 55), (width - 15, 55), (60, 60, 60), 1)

    y = 100
    row_h = 75

    # 1. SELECT — ring gauge, echoes a speedometer dial
    select_active = now < state["select_flash_until"]
    color = _icon_color(select_active)
    cv2.circle(panel, (60, y), 28, color, 3)
    cv2.circle(panel, (60, y), 6, color, -1)
    cv2.putText(panel, "SELECT", (110, y + 6), font, 0.55, color, 2)
    y += row_h

    # 2. PLAY / PAUSE
    color = ICON_COLOR
    if state["playing"]:
        pts = np.array([[45, y - 22], [45, y + 22], [80, y]], np.int32)
        cv2.fillPoly(panel, [pts], color)
    else:
        cv2.rectangle(panel, (42, y - 22), (55, y + 22), color, -1)
        cv2.rectangle(panel, (65, y - 22), (78, y + 22), color, -1)
    label = "PLAYING" if state["playing"] else "PAUSED"
    cv2.putText(panel, label, (110, y + 6), font, 0.55, color, 2)
    y += row_h

    # 3. VOLUME / MUTE — 5-bar equalizer-style gauge
    bars, filled = 5, round(state["volume_level"] / 10 * 5)
    for i in range(bars):
        bx, bh = 35 + i * 14, 14 + i * 8
        top = y + 22 - bh
        bar_color = (95, 95, 95) if state["muted"] or i >= filled else (0, 255, 180)
        cv2.rectangle(panel, (bx, top), (bx + 9, y + 22), bar_color, -1)
    if state["muted"]:
        cv2.line(panel, (30, y - 25), (100, y + 25), (0, 0, 255), 3)
    label = "MUTED" if state["muted"] else f"VOL {state['volume_level']}/10"
    label_color = (0, 0, 255) if state["muted"] else ICON_COLOR
    cv2.putText(panel, label, (110, y + 6), font, 0.55, label_color, 2)
    y += row_h

    # 4. TRACK NAV — left/right chevrons, echoes navigation arrows
    nav_left = state["nav_flash"] == "left" and now < state["nav_flash_until"]
    nav_right = state["nav_flash"] == "right" and now < state["nav_flash_until"]
    left_pts = np.array([[70, y - 20], [45, y], [70, y + 20]], np.int32)
    right_pts = np.array([[80, y - 20], [105, y], [80, y + 20]], np.int32)
    cv2.polylines(panel, [left_pts], False, _icon_color(nav_left), 4)
    cv2.polylines(panel, [right_pts], False, _icon_color(nav_right), 4)
    cv2.putText(panel, "TRACK NAV", (130, y + 6), font, 0.55, ICON_COLOR, 2)
    y += row_h

    # 5. SNAPSHOT
    snap_active = now < state["snapshot_flash_until"]
    color = _icon_color(snap_active)
    cv2.rectangle(panel, (35, y - 15), (85, y + 15), color, 3)
    cv2.circle(panel, (60, y), 10, color, 2)
    cv2.putText(panel, "SNAPSHOT", (110, y + 6), font, 0.55, color, 2)

    return panel


# ----------------------------------------------------------------------
# MAIN LOOP
# ----------------------------------------------------------------------
def main():
    global SYSTEM_CONTROL_ENABLED, SHOW_SIDE_PANEL

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    db = TelemetryDB()
    debouncer = DebounceStateMachine()
    swipe_detector = SwipeDetector()

    if SYSTEM_CONTROL_ENABLED and not PYAUTOGUI_AVAILABLE:
        print("[warning] SYSTEM_CONTROL_ENABLED is True but pyautogui is not installed. "
              "Run: pip install pyautogui")

    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam at index {CAM_INDEX}")

    fps_window = deque(maxlen=30)
    lighting_idx = 0
    benchmark_active = False
    benchmark_start_time = None
    benchmark_frames = 0
    benchmark_false_positives = 0
    benchmark_latencies = []

    hud_state = {
        "volume_level": 5,       # 0-10
        "muted": False,
        "playing": True,
        "nav_flash": None,       # 'left' / 'right'
        "nav_flash_until": 0.0,
        "snapshot_flash_until": 0.0,
        "select_flash_until": 0.0,
    }
    FLASH_DURATION_SEC = 0.4

    with mp_hands.Hands(
        model_complexity=0,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    ) as hands:

        try:
            while True:
                loop_start = time.perf_counter()

                # ---- Layer 1: capture ----
                t0 = time.perf_counter()
                ok, frame = cap.read()
                if not ok:
                    break
                frame = cv2.flip(frame, 1)  # selfie-mode
                capture_latency_ms = (time.perf_counter() - t0) * 1000

                # ---- Layer 2: inference ----
                t1 = time.perf_counter()
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)
                inference_latency_ms = (time.perf_counter() - t1) * 1000

                raw_gesture = None
                debug_info = None
                wrist_x_norm = None
                hand_present = bool(results.multi_hand_landmarks)
                if hand_present:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    if DRAW_LANDMARKS:
                        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    if DEBUG_GESTURE:
                        raw_gesture, debug_info = classify_gesture(hand_landmarks.landmark, debug=True)
                    else:
                        raw_gesture = classify_gesture(hand_landmarks.landmark)
                    wrist_x_norm = hand_landmarks.landmark[WRIST].x

                # ---- Layer 3: debounce (static hand-shape gestures) ----
                confirmed_gesture, just_triggered = debouncer.update(raw_gesture)

                # ---- Layer 3b: motion-based swipe (independent of the static debounce above) ----
                swipe_result = swipe_detector.update(wrist_x_norm, hand_present)
                if swipe_result:
                    confirmed_gesture = swipe_result
                    just_triggered = swipe_result

                total_latency_ms = (time.perf_counter() - loop_start) * 1000

                # ---- Layer 4: telemetry log ----
                db.log_frame(capture_latency_ms, inference_latency_ms, total_latency_ms, confirmed_gesture)

                # ---- Benchmark bookkeeping (only while 'b' toggled on) ----
                if benchmark_active:
                    benchmark_frames += 1
                    benchmark_latencies.append(total_latency_ms)
                    # A "false positive" here = a confirmed gesture firing with no
                    # hand in frame (defensive check; in practice raw_gesture is
                    # only ever set when a hand is detected).
                    if just_triggered and not hand_present:
                        benchmark_false_positives += 1

                # ---- Execute real system action on rising edge (only if enabled) ----
                if just_triggered and SYSTEM_CONTROL_ENABLED:
                    result = execute_system_action(just_triggered, frame)
                    if result:
                        print(f"[system action] {just_triggered} -> {result}")

                # ---- Update side-panel dashboard state on rising edge ----
                if just_triggered == "pinch":
                    hud_state["select_flash_until"] = loop_start + FLASH_DURATION_SEC
                elif just_triggered == "open_palm":
                    hud_state["playing"] = not hud_state["playing"]
                elif just_triggered == "point":
                    hud_state["muted"] = not hud_state["muted"]
                elif just_triggered == "thumbs_up":
                    hud_state["volume_level"] = min(10, hud_state["volume_level"] + 1)
                    hud_state["muted"] = False
                elif just_triggered == "thumbs_down":
                    hud_state["volume_level"] = max(0, hud_state["volume_level"] - 1)
                elif just_triggered == "peace":
                    hud_state["snapshot_flash_until"] = loop_start + FLASH_DURATION_SEC
                elif just_triggered in ("swipe_left", "swipe_right"):
                    hud_state["nav_flash"] = "left" if just_triggered == "swipe_left" else "right"
                    hud_state["nav_flash_until"] = loop_start + FLASH_DURATION_SEC

                # ---- FPS ----
                fps_window.append(1.0 / max(total_latency_ms / 1000, 1e-6))
                fps = sum(fps_window) / len(fps_window)

                # ---- Render ----
                lighting_label = LIGHTING_CONDITIONS[lighting_idx]
                frame = draw_hud(frame, confirmed_gesture, just_triggered, fps, total_latency_ms, lighting_label, debug_info)
                if benchmark_active:
                    cv2.putText(frame, "REC [BENCHMARK]", (frame.shape[1] - 260, frame.shape[0] - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                if SHOW_SIDE_PANEL:
                    panel = build_side_panel(PANEL_WIDTH, frame.shape[0], hud_state, loop_start)
                    display_frame = np.hstack([frame, panel])
                else:
                    display_frame = frame

                cv2.imshow("Gesture HUD", display_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('l'):
                    lighting_idx = (lighting_idx + 1) % len(LIGHTING_CONDITIONS)
                elif key == ord('x'):
                    SYSTEM_CONTROL_ENABLED = not SYSTEM_CONTROL_ENABLED
                    print(f"[system control] {'ENABLED - gestures will control your OS' if SYSTEM_CONTROL_ENABLED else 'disabled'}")
                elif key == ord('p'):
                    SHOW_SIDE_PANEL = not SHOW_SIDE_PANEL
                elif key == ord('b'):
                    if not benchmark_active:
                        benchmark_active = True
                        benchmark_start_time = datetime.now().isoformat(timespec="seconds")
                        benchmark_frames = 0
                        benchmark_false_positives = 0
                        benchmark_latencies = []
                    else:
                        benchmark_active = False
                        avg_latency = sum(benchmark_latencies) / len(benchmark_latencies) if benchmark_latencies else 0.0
                        db.write_benchmark_summary(
                            lighting_condition=lighting_label,
                            total_frames=benchmark_frames,
                            false_positives=benchmark_false_positives,
                            avg_latency_ms=avg_latency,
                            started_at=benchmark_start_time,
                            ended_at=datetime.now().isoformat(timespec="seconds"),
                        )
                        print(f"[benchmark] {lighting_label}: {benchmark_frames} frames, "
                              f"{benchmark_false_positives} false positives, "
                              f"avg latency {avg_latency:.2f} ms")

        finally:
            db.close()
            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
