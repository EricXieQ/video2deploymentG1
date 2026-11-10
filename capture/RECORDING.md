# Recording a motion for the G1

The robot can only copy what TRAM can see. Most problems with the dab's reference motion came from the capture: floating feet, a forward lean, and smoothing that flattens fast moves. A good recording saves days of fine-tuning.

## Any motion

- **30 fps**, landscape, 720p or 1080p. TRAM's VIMO and the retargeter both work at 30 fps. If the phone records at 60 fps, pass the true rate to the converter (`--fps 60`). A wrong label changes the speed: one early dab clip was labelled 60 fps by mistake and came out as a 2.3 s motion.
- **Tripod, level**, about waist to chest height, 3 to 4 m away. The converter assumes a level camera: it applies a fixed 90 degree turn from camera axes to "Z up".
- **Whole body in every frame, both feet included.** Without feet, TRAM guesses the ground contact, and the reference floats.
- **One person in frame.** TRAM keeps the longest person track (`hps_track_0`). Do not film anyone who has not agreed to it, and do not publish anyone's face without consent.
- **Start and end standing still**, in a stance close to the robot's walking stance: feet about hip-width apart, knees slightly bent, arms relaxed. The policy is handed over from the walking policy at the start and hands back at the end, so poses near that stance make both switches easier.
- **4 to 7 s long** (the dab was 5.93 s). Trim the video itself (for example `ffmpeg -ss 1.2 -t 6 -i in.mov -c copy out.mov`) rather than trimming later.
- Plain background, good light, fitted clothes.
- **Check:** `ffprobe -v error -select_streams v:0 -count_frames -show_entries stream=r_frame_rate,nb_read_frames in.mov` shows the frame rate and frame count.

## The "6-7" gesture

The gesture: palms up at chest height, hands alternately raised and lowered, often with a small knee bounce. Notes specific to it:

- **Keep it at or below about 2 hand cycles per second, driven mostly by the forearms (elbows).** Speed is not the limit: even 3 Hz uses under 25 % of the arm joints' 37 rad/s. Shoulder torque is the limit: a shoulder-driven 3 Hz swing of ±0.3 rad needs about 21 N m of inertia torque plus 5 N m of gravity, at the arm's 25 N m limit (training caps it near 20.6 N m). The shoulder PD loop's natural frequency is also only about 3.4 Hz (5.8 Hz at the elbow).
- **The capture pipeline is the bigger risk: it can smooth the oscillation away.** The converter's default 5-frame box filter keeps 96 % of a 1 Hz motion, 83 % at 2 Hz and only 65 % at 3 Hz. Use `--smooth 3` or `--smooth 0` for this gesture. The retargeter also smooths after every one of its 1000 steps; that effect is unmeasured.
- **Compare hand amplitude** in TRAM's `tram_output.mp4` with the retargeted motion before training (`vis_q_mj.py`, docs/PIPELINE.md step 4).
- **"Palms up" will not show on the robot.** The 23-DoF G1 config has no wrist joints (they are held at zero), and the retargeter zeroes SMPL's hands. The robot reproduces hand height and forearm motion, not palm orientation.
- **A small, deliberate bounce is fine; keep the lean slight.** Do not use the converter's `--lock_height` for this: a fixed pelvis with bending knees lifts the feet off the floor. Run `capture/ground_reference.py` after retargeting instead; with the lower foot pinned to the floor, the pelvis drops when the knees bend, which is what a bounce is.
- **Framing:** frontal to three-quarter view, so the up-down hand motion is not hidden behind the body.
- **Timing, about 6 s:** 1 s standing still in the walking-like stance, 4 s of gesture (6 to 8 alternations), then 1 s back to the start stance.
- At 2 Hz the robot's measured 50 ms command delay is 36 degrees of the cycle; training randomizes delay up to 60 ms for this reason (docs/PAPER_VS_THIS.md).

## Privacy

This repository publishes no videos and no faces. Keep phone videos local. The SMPL parameter file (`.npz`) that TRAM produces holds joint rotations and an average body shape, not images.
