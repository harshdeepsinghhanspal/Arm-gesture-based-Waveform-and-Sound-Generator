"""
Arm-waveform live synth
- Detects six points (L wrist, L elbow, L shoulder, R shoulder, R elbow, R wrist)
- Builds a wavetable from their vertical positions
- Plays sound using sounddevice with panning determined by body x-position
- Shows video with waveform drawn beneath the frame

Dependencies:
pip install mediapipe opencv-python numpy sounddevice
"""

import cv2
import mediapipe as mp
import numpy as np
import sounddevice as sd
import threading
import time

# --- Audio / synth parameters ---
SR = 44100                 # sample rate
BLOCKSIZE = 1024           # audio block size
WAVETABLE_SIZE = 4096      # resolution of wavetable (one cycle)
MIN_FREQ = 110.0           # Hz
MAX_FREQ = 880.0           # Hz
WAVEFORM_DISPLAY_H = 150   # pixels height of waveform display

# Shared state between video thread and audio callback
state_lock = threading.Lock()
wavetable = np.zeros(WAVETABLE_SIZE, dtype=np.float32)  # current waveform cycle
wavetable[:] = np.zeros_like(wavetable)                # start silent
current_freq = 220.0
current_pan = 0.5  # 0 = left, 1 = right
have_landmarks = False

# Phase accumulator for wavetable playback
phase = 0.0

# Initialize MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Helper functions

def landmarks_to_points(landmarks, image_w, image_h):
    """
    Return list of six (x,y) in pixel coords or None if missing.
    Order: L wrist, L elbow, L shoulder, R shoulder, R elbow, R wrist
    """
    idxs = {
        'LEFT_WRIST': mp_pose.PoseLandmark.LEFT_WRIST,
        'LEFT_ELBOW': mp_pose.PoseLandmark.LEFT_ELBOW,
        'LEFT_SHOULDER': mp_pose.PoseLandmark.LEFT_SHOULDER,
        'RIGHT_SHOULDER': mp_pose.PoseLandmark.RIGHT_SHOULDER,
        'RIGHT_ELBOW': mp_pose.PoseLandmark.RIGHT_ELBOW,
        'RIGHT_WRIST': mp_pose.PoseLandmark.RIGHT_WRIST,
    }
    order = ['LEFT_WRIST', 'LEFT_ELBOW', 'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'RIGHT_ELBOW', 'RIGHT_WRIST']
    pts = []
    for name in order:
        lm = landmarks.landmark[idxs[name]]
        # If visibility is low, treat as missing
        if lm.visibility is None or lm.visibility < 0.2:
            return None
        x_px = int(lm.x * image_w)
        y_px = int(lm.y * image_h)
        pts.append((x_px, y_px))
    return pts

def build_wavetable_from_points(points, img_h):
    """
    Given 6 points, convert their y-values to amplitude samples [-1,1],
    interpolate to WAVETABLE_SIZE and return wavetable.
    Higher y (lower on screen) -> more negative amplitude (convention).
    """
    # points is list of (x,y) in pixel coords
    ys = np.array([p[1] for p in points], dtype=np.float32)
    # Normalize: convert screen y -> [-1,1], with center of frame as 0
    # Using image height for scaling (img_h provided outside) but we can normalize relative to center
    center = img_h / 2.0
    amp = (center - ys) / center  # center->0, up -> positive, down -> negative
    # Create positions for those samples across [0,1]
    src_x = np.linspace(0.0, 1.0, len(amp))
    dst_x = np.linspace(0.0, 1.0, WAVETABLE_SIZE, endpoint=False)
    wav = np.interp(dst_x, src_x, amp).astype(np.float32)
    # Normalize to avoid clipping, apply smooth window to avoid hard edges
    wav = wav - np.mean(wav)
    max_abs = np.max(np.abs(wav)) if np.max(np.abs(wav)) > 0 else 1.0
    wav = wav / max_abs * 0.95
    # apply small smoothing (simple moving average) to reduce harsh steps
    kernel = np.ones(5) / 5.0
    wav = np.convolve(wav, kernel, mode='same').astype(np.float32)
    # Normalize again
    wav = wav / np.max(np.abs(wav) + 1e-9) * 0.95
    return wav

def pan_l_r(signal, pan):
    """
    Given mono signal array, return stereo 2-column array.
    pan: 0 => left only, 1 => right only. Use constant-power panning.
    """
    # constant-power panning
    angle = pan * np.pi/2.0
    left_gain = np.cos(angle)
    right_gain = np.sin(angle)
    stereo = np.empty((len(signal), 2), dtype=np.float32)
    stereo[:,0] = signal * left_gain
    stereo[:,1] = signal * right_gain
    return stereo

# Audio callback
def audio_callback(outdata, frames, time_info, status):
    global phase, wavetable, current_freq, current_pan
    if status:
        # If overflow/underflow, you can print or ignore
        print("Audio callback status:", status)
    t = np.arange(frames, dtype=np.float32)
    with state_lock:
        local_wav = wavetable.copy()
        freq = current_freq
        pan = current_pan
    if local_wav is None or np.all(local_wav == 0.0) or freq <= 0.0:
        outdata.fill(0)
        return
    # Phase increment per sample (index in wavetable)
    inc = (freq * WAVETABLE_SIZE) / SR
    # Prepare output mono buffer
    buf = np.zeros(frames, dtype=np.float32)
    # Fill buffer with wavetable lookup (linear interp)
    idxs = (phase + inc * np.arange(frames)) % WAVETABLE_SIZE
    idx_floor = np.floor(idxs).astype(int)
    frac = idxs - idx_floor
    next_idx = (idx_floor + 1) % WAVETABLE_SIZE
    buf = (1.0 - frac) * local_wav[idx_floor] + frac * local_wav[next_idx]
    phase = (phase + inc * frames) % WAVETABLE_SIZE
    # Apply amplitude envelope or smoothing if needed (here we don't)
    # Convert to stereo by panning
    stereo = pan_l_r(buf, pan)
    # Write to outdata
    outdata[:] = stereo.reshape(outdata.shape)

# Start audio stream
stream = sd.OutputStream(
    samplerate=SR,
    blocksize=BLOCKSIZE,
    dtype='float32',
    channels=2,
    callback=audio_callback
)

# Start audio stream in separate thread/context
stream.start()

# --- Video loop ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

prev_time = time.time()
while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame capture failed; exiting")
        break
    img_h, img_w = frame.shape[:2]
    # Flip horizontally for mirror view
    frame_rgb = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    frame_display = cv2.flip(frame, 1).copy()  # flip back for consistent coords

    points = None
    if results.pose_landmarks:
        pts = landmarks_to_points(results.pose_landmarks, img_w, img_h)
        if pts is not None:
            points = pts
            # draw points and connecting polyline
            for (x,y) in points:
                cv2.circle(frame_display, (x,y), 6, (0,255,0), -1)
            # draw polyline across the six points
            cv2.polylines(frame_display, [np.array(points, dtype=np.int32)], isClosed=False, color=(0,200,255), thickness=2)

            # compute arm span for frequency mapping: distance between wrists horizontally
            lwx, lwy = points[0]
            rwx, rwy = points[-1]
            wrist_span = abs(rwx - lwx)
            # map wrist_span (0..img_w) to frequency (MIN_FREQ..MAX_FREQ)
            span_norm = np.clip(wrist_span / float(img_w), 0.01, 1.0)
            freq = MIN_FREQ + (MAX_FREQ - MIN_FREQ) * span_norm

            # compute body midpoint x from shoulders for panning
            lsx, _ = points[2]  # left shoulder
            rsx, _ = points[3]  # right shoulder
            shoulders_mid_x = (lsx + rsx) / 2.0
            pan = np.clip(shoulders_mid_x / float(img_w), 0.0, 1.0)  # 0..1

            # build wavetable from these 6 points
            new_wav = build_wavetable_from_points(points, img_h)

            # Update shared state
            with state_lock:
                wavetable = new_wav
                current_freq = float(freq)
                current_pan = float(pan)
                have_landmarks = True

            # draw text info
            cv2.putText(frame_display, f"Freq: {freq:.1f} Hz  Pan: {pan:.2f}", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        else:
            # Missing some landmarks -> mark as not found
            with state_lock:
                have_landmarks = False
    else:
        with state_lock:
            have_landmarks = False

    # Draw waveform visualization under frame
    # Create blank area
    waveform_img = np.zeros((WAVEFORM_DISPLAY_H, img_w, 3), dtype=np.uint8)
    with state_lock:
        local_wav = wavetable.copy()
        local_have = have_landmarks
    if local_wav is not None:
        # Display a full cycle across width
        # downsample wavetable to image width
        x = np.linspace(0, WAVETABLE_SIZE, img_w, endpoint=False).astype(int) % WAVETABLE_SIZE
        samples = local_wav[x]
        # scale to pixel coordinates (center of waveform_img is mid line)
        mid = WAVEFORM_DISPLAY_H // 2
        amplitude = int(WAVEFORM_DISPLAY_H * 0.45)
        ys = (mid - (samples * amplitude)).astype(int)
        pts = np.column_stack((np.arange(img_w), ys))
        # background depending on landmark presence
        bgcolor = (30, 30, 30) if local_have else (10, 10, 10)
        waveform_img[:] = bgcolor
        # draw center line
        cv2.line(waveform_img, (0, mid), (img_w, mid), (80,80,80), 1)
        # draw waveform polyline
        cv2.polylines(waveform_img, [pts.astype(np.int32)], isClosed=False, color=(0,255,0), thickness=2)
        # draw filled area under curve for nicer look
        poly_pts = np.vstack([pts, [img_w-1, mid], [0, mid]]).astype(np.int32)
        cv2.fillPoly(waveform_img, [poly_pts], (0,50,0))
        # annotate
        status_text = "Waveform (from arms)" if local_have else "No pose detected - waveform frozen"
        cv2.putText(waveform_img, status_text, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220,220,220), 1)

    # Stack frame_display and waveform_img vertically
    combined = np.vstack((frame_display, waveform_img))

    # Show window
    cv2.imshow("Arm Waveform Synth (press q to quit)", combined)

    # handle key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
stream.stop()
stream.close()
pose.close()