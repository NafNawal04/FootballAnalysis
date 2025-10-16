from utils import read_video, save_video, get_center_of_bbox
from trackers import Tracker
import cv2
import numpy as np
from team_assigner import TeamAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator
from pass_and_interception_detector import PassAndInterceptionDetector
from drawers.pass_and_interceptions_drawer import PassInterceptionDrawer
from ball_acquisition import BallAcquisitionDetector
from court_keypoint_detector import CourtKeypointDetector
from tactical_view_converter import TacticalViewConverter
from drawers.tactical_view_drawer import TacticalViewDrawer


def main():
    
    # Read Video
    video_frames = read_video('input_videos/08fd33_4.mp4')

    # Initialize Tracker
    tracker = Tracker('models/football_player_detector.pt')

    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                       stub_path='stubs/track_stubs.pkl')
    # Get object positions 
    tracker.add_position_to_tracks(tracks)
    
    # Detect Court Keypoints
    court_keypoint_detector = CourtKeypointDetector('models/football_keypoint_detector.pt')
    court_keypoints = court_keypoint_detector.get_court_keypoints(video_frames,
                                                                 read_from_stub=True,
                                                                 stub_path='stubs/court_keypoints_stub.pkl')
    
    # Debug: Print keypoint detection results
    print(f"\nCourt Keypoint Detection Results:")
    print(f"Total frames: {len(court_keypoints)}")
    valid_keypoints = 0
    sample_keypoints = None
    for i, kp in enumerate(court_keypoints):
        if kp is not None and hasattr(kp, 'xy') and kp.xy is not None:
            try:
                kp_list = kp.xy.tolist()
                if len(kp_list) > 0 and len(kp_list[0]) > 0:
                    valid_keypoints += 1
                    if sample_keypoints is None:  # Get first valid keypoint as sample
                        sample_keypoints = kp_list[0]
                        print(f"Sample keypoints from frame {i}: {len(kp_list[0])} points")
                        for j, point in enumerate(kp_list[0][:5]):  # Show first 5 points
                            print(f"  Point {j}: ({point[0]:.1f}, {point[1]:.1f})")
            except Exception as e:
                print(f"Error processing keypoints in frame {i}: {e}")
    print(f"Frames with valid keypoints: {valid_keypoints}")

    # camera movement estimator
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
                                                                                read_from_stub=True,
                                                                                stub_path='stubs/camera_movement_stub.pkl')
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks,camera_movement_per_frame)


    # View Transformer
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)
    
    # Tactical View Converter
    tactical_view_converter = TacticalViewConverter('football_field.png')
    
    # Only proceed with tactical view if we have valid keypoints
    if valid_keypoints > 0:
        tactical_player_positions = tactical_view_converter.transform_players_to_tactical_view(court_keypoints, tracks['players'])
        print(f"Tactical view transformation completed")
        
        # Count how many frames have tactical positions
        frames_with_tactical = sum(1 for pos in tactical_player_positions if len(pos) > 0)
        print(f"Frames with tactical player positions: {frames_with_tactical}/{len(tactical_player_positions)}")
    else:
        print("No valid keypoints detected, creating fallback tactical view")
        # Create a simple fallback tactical view using relative positioning
        tactical_player_positions = []
        for frame_tracks in tracks['players']:
            tactical_positions = {}
            for player_id, player_data in frame_tracks.items():
                bbox = player_data["bbox"]
                # Simple relative positioning based on bbox center
                center_x, center_y = get_center_of_bbox(bbox)
                # Map to tactical view coordinates (simple scaling)
                tactical_x = (center_x / video_frames[0].shape[1]) * tactical_view_converter.width
                tactical_y = (center_y / video_frames[0].shape[0]) * tactical_view_converter.height
                tactical_positions[player_id] = [tactical_x, tactical_y]
            tactical_player_positions.append(tactical_positions)

    # Interpolate Ball Positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # Speed and distance estimator
    speed_and_distance_estimator = SpeedAndDistance_Estimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    # Assign Player Teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], 
                                    tracks['players'][0])
    
    # Add debug dictionary to collect team assignments
    team_assignments = {1: [], 2: []}  # Only 2 teams: white and mint
    
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num],   
                                                 track['bbox'],
                                                 player_id)
            tracks['players'][frame_num][player_id]['team'] = team 
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]
            
            # Collect player IDs by team
            if player_id not in team_assignments[team]:
                team_assignments[team].append(player_id)
    
    print("\nFinal Team Assignments:")
    print(f"Team 1 (White): Players: {sorted(team_assignments[1])}")
    print(f"Team 2 (Mint): Players: {sorted(team_assignments[2])}")
    
    # Detect Ball Acquisition using improved algorithm
    ball_acquisition_detector = BallAcquisitionDetector()
    ball_acquisition = ball_acquisition_detector.detect_ball_possession(tracks['players'], tracks['ball'])
    
    # Create team ball control and player assignment data
    team_ball_control = []
    player_assignment = []  # Track team assignments for each frame
    
    for frame_num, player_track in enumerate(tracks['players']):
        assigned_player = ball_acquisition[frame_num] if frame_num < len(ball_acquisition) else -1
        
        if assigned_player != -1:
            # Mark player as having ball
            if assigned_player in player_track:
                tracks['players'][frame_num][assigned_player]['has_ball'] = True
                team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
            else:
                # Fallback if player not found in current frame
                if len(team_ball_control) > 0:
                    team_ball_control.append(team_ball_control[-1])
                else:
                    team_ball_control.append(1)  # Default to team 1
        else:
            # No player has ball
            if len(team_ball_control) > 0:
                team_ball_control.append(team_ball_control[-1])
            else:
                team_ball_control.append(1)  # Default to team 1
        
        # Create player assignment dictionary for this frame
        frame_player_assignment = {}
        for player_id, track in player_track.items():
            frame_player_assignment[player_id] = track.get('team', -1)
        player_assignment.append(frame_player_assignment)
    
    team_ball_control = np.array(team_ball_control)
    
    # Detect Passes and Interceptions
    pass_interception_detector = PassAndInterceptionDetector()
    passes = pass_interception_detector.detect_passes(ball_acquisition, player_assignment)
    interceptions = pass_interception_detector.detect_interceptions(ball_acquisition, player_assignment)
    
    print(f"\nPass and Interception Summary:")
    print(f"Total passes detected: {len([p for p in passes if p != -1])}")
    print(f"Total interceptions detected: {len([i for i in interceptions if i != -1])}")


    # Draw output 
    ## Draw object Tracks
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)

    ## Draw Camera movement
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames, camera_movement_per_frame)

    ## Draw Speed and Distance
    speed_and_distance_estimator.draw_speed_and_distance(output_video_frames, tracks)
    
    ## Draw Pass and Interception Statistics
    pass_interception_drawer = PassInterceptionDrawer()
    pass_interception_drawer.clear_cache()
    output_video_frames = pass_interception_drawer.draw(output_video_frames, passes, interceptions,
                                                         ball_acquisition,player_assignment, tracks)
    
    
    tactical_view_drawer = TacticalViewDrawer()
    output_video_frames = tactical_view_drawer.draw(output_video_frames,
                                                   'football_field.png',
                                                   tactical_view_converter.width,
                                                   tactical_view_converter.height,
                                                   tactical_view_converter.key_points,
                                                   tactical_player_positions,
                                                   player_assignment,
                                                   ball_acquisition)

    # Save video
    save_video(output_video_frames, 'output_videos/output_video7.avi')

if __name__ == '__main__':
    
    main()