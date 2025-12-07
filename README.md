# 🎵 Arm-Waveform Live Synth

A real-time pose-controlled audio synthesizer that turns your arm movements into sound waves!
Using a webcam and MediaPipe pose tracking, the script builds a wavetable from your arm positions and plays it live with panning and frequency control. 🎧✨

#🔧 Features:
> 🧍‍♂️ Pose Tracking (6 Key Arm Points)
Detects: Left wrist, left elbow, left shoulder, right shoulder, right elbow, right wrist.

> 🎚️ Dynamic Frequency Control
Frequency changes based on the horizontal distance between wrists.

> 🎛️ Stereo Panning
Panning dynamically shifts left ↔ right depending on shoulder midpoint.

> 🎨 Live Waveform Visualization
Displays the synthesized waveform below the webcam feed.

> 🔊 Real-Time Wavetable Synth
Generates audio using your arm positions as waveform samples.

#📦 Dependencies:
> pip install mediapipe opencv-python numpy sounddevice

#▶️ How to Run:
> Use this command: python Wave_Generator.py
Make sure you have a webcam connected.
Press Q anytime to quit the program.

#🎮 How It Works (In Short)
> ✋ Raise or move your arms → waveform shape changes
> 🤲 Move wrists apart → frequency increases
> 🧍 Lean left/right → audio pans to that side
> 👀 Watch the live waveform visualization update in sync
