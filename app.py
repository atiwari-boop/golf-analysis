import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import os
from moviepy.editor import ImageSequenceClip
import ffmpeg

# Function to detect the rotation angle of the image
def  get_video_rotation ( video_path ):
    try :
        probe = ffmpeg.probe(video_path)
        video_stream = next ((stream for stream in probe[ 'streams' ] if stream[ 'codec_type' ] == 'video' ), None )
        if video_stream and  'tags'  in video_stream and  'rotate'  in video_stream[ 'tags' ]:
            return  int (video_stream[ 'tags' ][ 'rotate' ])
    except ffmpeg.Error as e:
        st.warning( f"Failed to read rotation information of image: {e.stderr} " )
    return  0

# -------------------------------------------------------------------
#1. Final Analysis Engine
# -------------------------------------------------------------------
def  analyze_swing ( video_path, progress_bar ):
    rotation_angle = get_video_rotation(video_path)
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence= 0.5 , min_tracking_confidence= 0.7 )
    mp_drawing = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(video_path)

    if  not cap.isOpened():
        st.error( "Failed to open video." )
        return  None , None , None
            
    total_frames = int (cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0 :
        st.error( "Unable to read the number of frames in the video." )
        return  None , None , None

    original_width = int (cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int (cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    is_vertical = (rotation_angle in [ 90 , 270 ]) or \
                  (rotation_angle == 0  and original_height > original_width)
    
    if is_vertical:
        display_height = 1280
        display_width = 720
    else :
        display_height = 720
        display_width = 1280

    output_frames = []
    window_size = 7
    landmark_buffers = {i: [] for i in  range ( 33 )}
    swing_plane_line = None
    min_y_coord = 9999
    top_position_coords = None
    address_wrist_y = 0
    backswing_height = 0
    impact_hand_y = 0
    top_detected = False
    right_hand_path = []
    # New data series for advanced analysis
    raw_shoulder_angles = []  # absolute screen-space angles each frame
    raw_hip_angles = []       # absolute screen-space angles each frame
    shoulder_rotations = []   # relative (computed post-loop)
    hip_rotations = []        # relative (computed post-loop)
    x_factor_series = []      # computed post-loop
    wrist_y_series = []       # wrist y in pixels (for tempo detection)
    head_positions = []      # pixel positions of nose for heatmap & movement metric
    address_frame_index = None
    address_torso_height_px = None  # shoulder-hip vertical distance at address for normalization
    top_frame_index = None
    impact_frame_index = None
    address_vectors = {}
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 : fps = 30.0

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if  not ret:
            break
        
        if rotation_angle == 90 :
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif rotation_angle == 180 :
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif rotation_angle == 270 :
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        frame_count += 1
        if progress_bar:
            progress_percentage = frame_count / total_frames
            progress_bar.progress(progress_percentage, text= f"Analyzing swing... { int (progress_percentage * 100 )} %" )

        resized_frame = cv2.resize(frame, (display_width, display_height))
        image_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = pose.process(image_rgb)
        image_rgb.flags.writeable = True

        if results.pose_landmarks:
            for i, landmark in  enumerate (results.pose_landmarks.landmark):
                buffer = landmark_buffers[i]
                buffer.append([landmark.x, landmark.y, landmark.z, landmark.visibility])
                if  len (buffer) > window_size:
                    buffer.pop( 0 )
                smoothed_point = np.mean(buffer, axis= 0 )
                landmark.x, landmark.y, landmark.z, landmark.visibility = smoothed_point
            
            landmarks = results.pose_landmarks.landmark
            right_wrist_landmark = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
            left_wrist_landmark = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
            l_sh = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            r_sh = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            l_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
            r_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
            nose = landmarks[mp_pose.PoseLandmark.NOSE]

            # Pixel head position for heatmap later
            head_positions.append((int(nose.x * display_width), int(nose.y * display_height)))

            # Compute horizontal rotation proxies using shoulder and hip line orientation (2D screen space)
            def line_angle(a, b):
                return np.degrees(np.arctan2(b.y - a.y, b.x - a.x))

            shoulder_angle = line_angle(r_sh, l_sh)
            hip_angle = line_angle(r_hip, l_hip)
            raw_shoulder_angles.append(shoulder_angle)
            raw_hip_angles.append(hip_angle)

            if address_frame_index is None and right_wrist_landmark.visibility > 0.6:
                address_frame_index = frame_count
                address_vectors['shoulder_angle'] = shoulder_angle
                address_vectors['hip_angle'] = hip_angle
                # Compute torso height in pixels (vertical mid-shoulder to mid-hip distance)
                mid_sh_y = ((l_sh.y + r_sh.y)/2) * display_height
                mid_hip_y = ((l_hip.y + r_hip.y)/2) * display_height
                address_torso_height_px = abs(mid_hip_y - mid_sh_y)

            # Relative rotations deferred to post-loop for cleaner unwrapping

            if right_wrist_landmark.visibility > 0.7 :
                if swing_plane_line is  None :
                    shoulder_x = (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x + landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x) / 2
                    shoulder_y = (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y + landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y) / 2
                    wrist_x = (landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x + right_wrist_landmark.x) / 2
                    wrist_y = (landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y + right_wrist_landmark.y) / 2
                    p1 = ( int (wrist_x * display_width), int (wrist_y * display_height))
                    p2 = ( int (shoulder_x * display_width), int (shoulder_y * display_height))
                    swing_plane_line = (p1, p2)
                    address_wrist_y = p1[ 1 ]
                
                right_wrist_coord = ( int (right_wrist_landmark.x * display_width), int (right_wrist_landmark.y * display_height))
                right_hand_path.append(right_wrist_coord)
                wrist_y_series.append(right_wrist_coord[1])

                if right_wrist_coord[ 1 ] < min_y_coord:
                    min_y_coord = right_wrist_coord[ 1 ]
                    top_position_coords = right_wrist_coord
                    top_detected = True
                    if top_frame_index is None:
                        top_frame_index = frame_count
                    if address_wrist_y > 0 :
                        backswing_height = address_wrist_y - min_y_coord
                
                if top_detected and impact_hand_y == 0  and right_wrist_coord[ 1 ] >= address_wrist_y * 0.95 :
                    impact_hand_y = right_wrist_coord[ 1 ]
                    if impact_frame_index is None:
                        impact_frame_index = frame_count

        text = f"Backswing Height: {backswing_height} px"
        cv2.putText(resized_frame, text, ( 20 , 60 ), cv2.FONT_HERSHEY_SIMPLEX, 1.5 , ( 255 , 255 , 255 ), 3 , cv2.LINE_AA)
        if swing_plane_line:
            cv2.line(resized_frame, swing_plane_line[ 0 ], swing_plane_line[ 1 ], color=( 255 , 0 , 0 ), thickness= 3 )
        if  len (right_hand_path) > 1 :
            cv2.polylines(resized_frame, [np.array(right_hand_path)], isClosed= False , color=( 0 , 255 , 0 ), thickness= 4 )
        if top_position_coords:
            cv2.circle(resized_frame, top_position_coords, 15 , ( 0 , 255 , 255 ), 4 )
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image=resized_frame, landmark_list=results.pose_landmarks, connections=mp_pose.POSE_CONNECTIONS)
        
        output_frames.append(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB))

    cap.release()
    # --- Post-loop processing ---
    def unwrap(series):
        if not series: return []
        rad = np.radians(series)
        unwrapped = np.unwrap(rad)
        return list(np.degrees(unwrapped))

    unwrapped_shoulder = unwrap(raw_shoulder_angles)
    unwrapped_hip = unwrap(raw_hip_angles)

    # Use address frame index to define baseline; otherwise first frame
    if unwrapped_shoulder:
        base_idx = address_frame_index if address_frame_index is not None and address_frame_index < len(unwrapped_shoulder) else 0
        base_sh = unwrapped_shoulder[base_idx]
        base_hip = unwrapped_hip[base_idx] if base_idx < len(unwrapped_hip) else unwrapped_hip[0]
        for sh, hp in zip(unwrapped_shoulder, unwrapped_hip):
            # Signed shortest difference relative to address baseline
            rel_sh = ( (sh - base_sh + 180) % 360 ) - 180
            rel_hip = ( (hp - base_hip + 180) % 360 ) - 180
            # Constrain to plausible ranges (-150, 150)
            rel_sh = float(np.clip(rel_sh, -150, 150))
            rel_hip = float(np.clip(rel_hip, -120, 120))
            shoulder_rotations.append(rel_sh)
            hip_rotations.append(rel_hip)
            x_factor_series.append(rel_sh - rel_hip)

    # Clamp X-factor to plausible biomechanical range (-50, 90)
    x_factor_series = [float(np.clip(xf, -50, 90)) for xf in x_factor_series]

    def safe_stats(arr):
        if not arr:
            return {"max":0,"min":0,"range":0,"avg":0}
        a = np.array(arr, dtype=float)
        return {"max":float(np.max(a)),"min":float(np.min(a)),"range":float(np.max(a)-np.min(a)),"avg":float(np.mean(a))}

    shoulder_stats = safe_stats(shoulder_rotations)
    hip_stats = safe_stats(hip_rotations)
    xfactor_stats = safe_stats(x_factor_series)

    # Improved tempo detection using wrist Y series if frame indices failed
    tempo_ratio = None
    if top_frame_index is None and wrist_y_series:
        # Top = min y (highest point) index
        top_frame_index = int(np.argmin(wrist_y_series))
    if impact_frame_index is None and wrist_y_series and top_frame_index is not None:
        # Impact ~ first point after top where wrist within 5% of initial wrist y
        if wrist_y_series:
            initial_y = wrist_y_series[0]
            for idx in range(top_frame_index+1, len(wrist_y_series)):
                if abs(wrist_y_series[idx] - initial_y) <= initial_y * 0.05:
                    impact_frame_index = idx
                    break
    if address_frame_index is None and wrist_y_series:
        address_frame_index = 0
    if address_frame_index is not None and top_frame_index is not None and impact_frame_index is not None and impact_frame_index > top_frame_index > address_frame_index:
        backswing_time = (top_frame_index - address_frame_index)/fps
        downswing_time = (impact_frame_index - top_frame_index)/fps
        if downswing_time > 0.0 and backswing_time > 0.1:
            tempo_ratio = backswing_time / downswing_time
            if tempo_ratio < 0 or tempo_ratio > 8:
                tempo_ratio = None

    # Head movement metric (max displacement from first recorded head position)
    head_movement_px = 0
    if len(head_positions) > 3:
        hx0, hy0 = head_positions[0]
        head_movement_px = max(int(np.hypot(hx - hx0, hy - hy0)) for hx, hy in head_positions)

    def valid_or_none(val, lo, hi):
        return round(val,1) if lo <= val <= hi else None
    # Normalize pixel-based metrics by torso height (% of address torso height)
    if address_torso_height_px and address_torso_height_px > 0:
        backswing_height_norm = round(100 * backswing_height / address_torso_height_px, 1)
        head_movement_norm = round(100 * head_movement_px / address_torso_height_px, 1)
    else:
        backswing_height_norm = backswing_height
        head_movement_norm = head_movement_px

    analysis_results = {
        "max_height" : backswing_height_norm,
        "impact_height_vs_address" : impact_hand_y - address_wrist_y if impact_hand_y > 0  else  0,
    "tempo_ratio": round(tempo_ratio,2) if tempo_ratio is not None else None,
        "shoulder_rotation_max": valid_or_none(shoulder_stats['max'], -10, 150),
        "shoulder_rotation_range": valid_or_none(shoulder_stats['range'], 0, 160),
        "hip_rotation_max": valid_or_none(hip_stats['max'], -10, 120),
        "x_factor_max": valid_or_none(xfactor_stats['max'], -50, 90),
        "x_factor_range": valid_or_none(xfactor_stats['range'], 0, 140),
        "head_movement_px": head_movement_norm,
        "torso_height_px": round(address_torso_height_px,1) if address_torso_height_px else None,
        "backswing_height_raw_px": backswing_height,
        "head_movement_raw_px": head_movement_px
    }
    if not shoulder_rotations:
        analysis_results['note'] = 'Rotation metrics not available (insufficient landmark confidence early in swing).'
    else:
        # Flag if any value became None due to plausibility check
        if any(analysis_results[k] is None for k in ["shoulder_rotation_max","hip_rotation_max","x_factor_max"]):
            analysis_results['warning'] = 'Some rotation metrics discarded as implausible (camera angle or detection error).'

    fig = None
    if right_hand_path:
        hand_heights = [p[ 1 ] for p in right_hand_path]
        hand_heights_inverted = [display_height - h for h in hand_heights]
        fig, ax = plt.subplots(figsize=( 10 , 6 ))
        ax.plot(hand_heights_inverted, color= 'green' , label= 'Hand Height' )
        ax.set_title( 'Hand Height Over Time' )
        ax.set_xlabel( 'Frame Number' )
        ax.set_ylabel( 'Hand Height (pixels from bottom)' )
        ax.legend()
        ax.grid( True )

    # Additional figures dictionary
    extra_figs = {}
    if shoulder_rotations and hip_rotations:
        rot_fig, rax = plt.subplots(figsize=(8,4))
        rax.plot(shoulder_rotations, label='Shoulders (°)', color='tab:blue')
        rax.plot(hip_rotations, label='Hips (°)', color='tab:orange')
        if x_factor_series:
            rax.plot(x_factor_series, label='X-Factor (°)', color='tab:green', alpha=0.6)
        if top_frame_index:
            rax.axvline(top_frame_index, color='purple', linestyle='--', label='Top')
        if impact_frame_index:
            rax.axvline(impact_frame_index, color='red', linestyle='--', label='Impact')
        rax.set_title('Rotation Angles Over Time')
        rax.set_xlabel('Frame')
        rax.set_ylabel('Degrees vs Address')
        rax.legend()
        rax.grid(True, alpha=0.3)
        extra_figs['rotation'] = rot_fig

    if x_factor_series:
        xf_fig, xf_ax = plt.subplots(figsize=(8,3))
        xf_ax.plot(x_factor_series, color='tab:green')
        if top_frame_index:
            xf_ax.axvline(top_frame_index, color='purple', linestyle='--', label='Top')
        if impact_frame_index:
            xf_ax.axvline(impact_frame_index, color='red', linestyle='--', label='Impact')
        xf_ax.set_title('X-Factor (Shoulder - Hip Rotation)')
        xf_ax.set_xlabel('Frame')
        xf_ax.set_ylabel('Degrees')
        xf_ax.grid(True, alpha=0.3)
        xf_ax.legend()
        extra_figs['xfactor'] = xf_fig

    if head_positions:
        heat_fig, hax = plt.subplots(figsize=(5,5))
        xs = [p[0] for p in head_positions]
        ys = [p[1] for p in head_positions]
        heatmap, xedges, yedges = np.histogram2d(xs, ys, bins=40, range=[[0, display_width],[0, display_height]])
        # Flip vertically for display alignment
        heatmap = np.flipud(heatmap)
        hax.imshow(heatmap, cmap='hot', extent=[0, display_width, 0, display_height], aspect='auto')
        hax.set_title('Head Motion Heatmap')
        hax.set_xlabel('X (px)')
        hax.set_ylabel('Y (px)')
        extra_figs['head_heatmap'] = heat_fig

    output_video_path = None
    if output_frames:
        with tempfile.NamedTemporaryFile(delete= False , suffix= '.mp4' ) as tfile:
            output_video_path = tfile.name
        clip = ImageSequenceClip(output_frames, fps=fps)
        # --- [Last modified part 1] ---
        # Change to use 'libx264', the web browser standard codec
        clip.write_videofile(output_video_path, codec= 'libx264' , logger= None )

    # --- Posture & Swing Evaluation Heuristics ---
    def classify(metric, good_range=None, max_ok=None, target=None, tol=0):
        if metric is None: return 'unknown'
        if good_range:
            lo, hi = good_range
            return 'good' if lo <= metric <= hi else 'improve'
        if target is not None:
            return 'good' if abs(metric - target) <= tol else 'improve'
        if max_ok is not None:
            return 'good' if metric <= max_ok else 'improve'
        return 'unknown'

    eval_items = []
    # Tempo (ideal about 3:1)
    tempo_val = analysis_results['tempo_ratio']
    tempo_class = classify(tempo_val, target=3.0, tol=0.6)
    eval_items.append(("Tempo", tempo_val, tempo_class, "Aim near 3.0 (backswing:downswing)"))
    # Shoulder rotation (typical full swing 70–100 screen-proxy)
    sh_max = analysis_results['shoulder_rotation_max']
    if sh_max is not None:
        eval_items.append(("Shoulder Rotation", sh_max, 'good' if sh_max >= 70 else 'improve', "Greater shoulder turn can add coil"))
    # Hip rotation (typical 35–55 at top)
    hip_max = analysis_results['hip_rotation_max']
    if hip_max is not None:
        hip_class = classify(hip_max, good_range=(30,60))
        eval_items.append(("Hip Rotation", hip_max, hip_class, "Maintain stable but not locked hips"))
    # X-Factor max (30–55 common; >65 maybe excessive separation)
    xf_max = analysis_results['x_factor_max']
    if xf_max is not None:
        xf_class = classify(xf_max, good_range=(28,60))
        eval_items.append(("X-Factor", xf_max, xf_class, "Efficient separation aids power"))
    # Head movement percentage of torso height at address (<10% good, 10–15% watch, >15% improve)
    head_mv = analysis_results['head_movement_px']  # already normalized (% torso) if torso height captured
    head_class = 'good' if head_mv < 10 else ('watch' if head_mv < 15 else 'improve')
    eval_items.append(("Head Stability (% torso)", head_mv, head_class, "Keep head steady (<10% torso movement ideal)"))
    # Impact hand height difference
    impact_diff = analysis_results['impact_height_vs_address']
    impact_class = 'good' if abs(impact_diff) <= 15 else 'improve'
    eval_items.append(("Impact Hand Height", impact_diff, impact_class, "Maintain original posture through impact"))

    positives = []
    improvements = []
    watch_list = []
    for label, val, cls, note in eval_items:
        entry = f"{label}: {val} ({note})"
        if cls == 'good': positives.append(entry)
        elif cls == 'watch': watch_list.append(entry)
        elif cls == 'improve': improvements.append(entry)

    good_count = len(positives)
    improve_count = len(improvements)
    overall = 'Good' if improve_count == 0 and good_count >= 3 else ('Needs Improvement' if improve_count >= 2 else 'Mixed')

    evaluation = {
        'overall': overall,
        'positives': positives,
        'watch': watch_list,
        'improvements': improvements
    }

    return output_video_path, fig, analysis_results, extra_figs, evaluation

# -------------------------------------------------------------------
#2. Streamlit website UI code
# -------------------------------------------------------------------
st.set_page_config(layout= "wide" , page_title= "AI Golf Swing Analysis" )
st.title( "🏌️ AI Golf Swing Automatic Analysis Report" )

# Helper for displaying None values as an em dash
def safe_display(val):
    return "—" if val is None else val
st.info ( "It is equipped with an improved AI engine that compensates for the shaking of all joints and automatically recognizes and analyzes vertical images." )

uploaded_file = st.file_uploader(
    "Upload a video of your swing (both vertical and horizontal video is acceptable)" ,
    type =[ "mp4" , "mov" , "mpeg4" ]
)

if uploaded_file is  not  None :
    # --- [Last modified part 2] ---
    # Improved stability by passing the original video directly as bytes instead of as a file path.
    video_bytes = uploaded_file.getvalue()
    with tempfile.NamedTemporaryFile(delete= False , suffix= '.mp4' ) as tfile:
        tfile.write(video_bytes)
        temp_video_path = tfile.name

    st.header( "✅ Uploaded original video" )
    st.video(video_bytes) # Use byte data instead of file path

    if st.button( "Start analysis!" ):
        progress_bar = st.progress( 0 , text= "Starting swing analysis. Please wait a moment..." )
        result_video_path, result_graph, insights, extra_figs, evaluation = analyze_swing(temp_video_path, progress_bar)
        progress_bar.progress( 1.0 , text= "Analysis complete!" )

        if result_video_path:
            st.success( "Analysis completed successfully!" )
            col1, col2 = st.columns( 2 )
            with col1:
                st.header( "Analysis Results Video" )
                with  open (result_video_path, 'rb' ) as video_file:
                    result_video_bytes = video_file.read()
                st.video(result_video_bytes)
                # Overall evaluation summary
                st.subheader("Overall Assessment")
                color_map = { 'Good': st.success, 'Mixed': st.info, 'Needs Improvement': st.warning }
                color_map.get(evaluation['overall'], st.info)(f"Overall: {evaluation['overall']}")
                if evaluation['positives']:
                    st.markdown("**Strengths:**")
                    for p in evaluation['positives']:
                        st.write(f"✅ {p}")
                if evaluation['watch']:
                    st.markdown("**Monitor:**")
                    for w in evaluation['watch']:
                        st.write(f"👀 {w}")
                if evaluation['improvements']:
                    st.markdown("**Improvements:**")
                    for imp in evaluation['improvements']:
                        st.write(f"⚠️ {imp}")
            with col2:
                st.header("Graphs")
                gcol1, gcol2 = st.columns(2)
                with gcol1:
                    if result_graph is not None:
                        st.caption("Hand Height")
                        st.pyplot(result_graph)
                    if 'rotation' in extra_figs:
                        st.caption("Shoulder/Hip Rotation")
                        st.pyplot(extra_figs['rotation'])
                with gcol2:
                    if 'xfactor' in extra_figs:
                        st.caption("X-Factor")
                        st.pyplot(extra_figs['xfactor'])
                    if 'head_heatmap' in extra_figs:
                        st.caption("Head Motion Heatmap")
                        st.pyplot(extra_figs['head_heatmap'])

            st.header( "‍⚕️ Automatic Swing Diagnostic Report" )
            st.metric( "Maximum backswing height (relative to address)" , f" {insights[ 'max_height' ]} px" )
            impact_metric = insights[ 'impact_height_vs_address' ]
            st.metric( "Hand height at impact (relative to address)" , f" {impact_metric} px" )
            # Additional metrics row
            colm1, colm2, colm3, colm4 = st.columns(4)
            with colm1:
                tempo_val = insights.get('tempo_ratio', None)
                st.metric("Tempo Ratio (B:D)", "—" if tempo_val is None else tempo_val)
                st.metric(
                    "Head Move (% torso)",
                    insights.get('head_movement_px', 0),
                    help=f"Raw displacement: {insights.get('head_movement_raw_px', 'n/a')} px"
                )
            with colm2:
                st.metric("Shoulder Max Rot (°)", safe_display(insights.get('shoulder_rotation_max')))
                st.metric("Shoulder Range (°)", safe_display(insights.get('shoulder_rotation_range')))
            with colm3:
                st.metric("Hip Max Rot (°)", safe_display(insights.get('hip_rotation_max')))
                st.metric("X-Factor Max (°)", safe_display(insights.get('x_factor_max')))
            with colm4:
                st.metric("X-Factor Range (°)", safe_display(insights.get('x_factor_range')))

            if impact_metric <= 15 :
                st.info( "✔️ **Impact Position**: Good! Maintained a good impact height similar to the address." )
            else :
                st.warning( "⚠️ **Improvement**: Your hands tend to be higher than your address at impact. This could be a 'batch' motion where your upper body rises early, so check that out." )
        
        else :
            st.error( "Video analysis failed. Please try another video or make sure it is not too short." )