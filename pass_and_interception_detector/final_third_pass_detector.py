import numpy as np
from typing import Dict, List, Tuple, Optional

class FinalThirdPassDetector:
    """
    Detects final third passes in football matches.
    
    A final third pass is any pass that moves the ball from outside the attacking 
    third of the pitch into the opponent's final third. The pass is counted as 
    successful only if the next touch on the ball inside that zone is by a teammate.
    """
    
    def __init__(self, view_transformer, pitch_length=105, pitch_width=68):
        """
        Initialize the Final Third Pass Detector.
        
        Args:
            view_transformer: ViewTransformer instance for mapping coordinates to pitch
            pitch_length: Length of the pitch in meters (default: 105m)
            pitch_width: Width of the pitch in meters (default: 68m)
        """
        self.view_transformer = view_transformer
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width
        
        # Team attacking directions: 1 = 'right', 2 = 'left' (default assumption)
        # Can be updated based on actual game analysis
        self.team_directions = {
            1: 'right',  # Team 1 attacks to the right (higher x values)
            2: 'left'    # Team 2 attacks to the left (lower x values)
        }
    
    def set_attacking_direction(self, team_id: int, direction: str):
        """
        Set the attacking direction for a team.
        
        Args:
            team_id: Team identifier (1 or 2)
            direction: 'left' or 'right'
        """
        if direction not in ['left', 'right']:
            raise ValueError("Direction must be 'left' or 'right'")
        self.team_directions[team_id] = direction
    
    def _get_final_third_boundary(self, team_id: int) -> float:
        """
        Get the x-coordinate boundary for the final third for a given team.
        
        Args:
            team_id: Team identifier (1 or 2)
            
        Returns:
            x-coordinate marking the start of the final third
        """
        if self.team_directions[team_id] == 'right':
            # Attacking right: final third is x > 2/3 * pitch_length
            return (2.0 / 3.0) * self.pitch_length
        else:
            # Attacking left: final third is x < 1/3 * pitch_length
            return (1.0 / 3.0) * self.pitch_length
    
    def _is_in_final_third(self, x_coord: float, team_id: int) -> bool:
        """
        Check if a position is in the final third for a given team.
        
        Args:
            x_coord: X-coordinate on the pitch
            team_id: Team identifier (1 or 2)
            
        Returns:
            True if position is in the attacking final third for the team
        """
        boundary = self._get_final_third_boundary(team_id)
        
        if self.team_directions[team_id] == 'right':
            return x_coord > boundary
        else:
            return x_coord < boundary
    
    def _get_ball_position_at_frame(self, ball_tracks: List[Dict], frame_num: int) -> Optional[np.ndarray]:
        """
        Get the transformed ball position at a specific frame.
        
        Args:
            ball_tracks: Ball tracking data
            frame_num: Frame number
            
        Returns:
            Transformed position [x, y] or None if not available
        """
        if frame_num >= len(ball_tracks) or frame_num < 0:
            return None
        
        frame_data = ball_tracks[frame_num]
        if not frame_data or 1 not in frame_data:
            return None
        
        ball_info = frame_data[1]
        
        # Try to get transformed position first
        if 'position_transformed' in ball_info:
            pos = ball_info['position_transformed']
            if pos is not None and len(pos) == 2:
                return np.array(pos)
        
        # Fallback to position_adjusted and transform it
        if 'position_adjusted' in ball_info:
            pos = ball_info['position_adjusted']
            if pos is not None and len(pos) == 2:
                pos_array = np.array([pos])
                transformed = self.view_transformer.transform_points(pos_array)
                if transformed is not None and len(transformed) > 0:
                    return transformed[0]
        
        return None
    
    def detect_final_third_passes(
        self, 
        passes: List[int], 
        ball_acquisition: List[int], 
        player_assignment: List[Dict[int, int]], 
        ball_tracks: List[Dict],
        player_tracks: List[Dict]
    ) -> Tuple[List[int], Dict]:
        """
        Detect final third passes from existing pass data.
        
        Args:
            passes: List indicating successful passes (frame -> team_id or -1)
            ball_acquisition: List indicating ball possession (frame -> player_id or -1)
            player_assignment: List of dicts mapping player_id -> team_id per frame
            ball_tracks: Ball tracking data with positions
            player_tracks: Player tracking data
            
        Returns:
            Tuple of:
                - List indicating final third passes (frame -> team_id or -1)
                - Dictionary with detailed pass information
        """
        final_third_passes = [-1] * len(passes)
        pass_details = []
        
        prev_holder = -1
        prev_holder_frame = -1
        prev_ball_pos = None
        
        for frame in range(len(ball_acquisition)):
            current_holder = ball_acquisition[frame]
            
            # When ball changes hands
            if current_holder != -1:
                if prev_holder != -1 and prev_holder != current_holder:
                    # Check if this is a successful pass
                    if frame < len(passes) and passes[frame] != -1:
                        # Get team information
                        if (prev_holder_frame < len(player_assignment) and 
                            frame < len(player_assignment)):
                            
                            prev_team = player_assignment[prev_holder_frame].get(prev_holder, -1)
                            current_team = player_assignment[frame].get(current_holder, -1)
                            
                            # Verify it's a same-team pass
                            if prev_team == current_team and prev_team != -1:
                                # Get ball positions at pass start and end
                                start_pos = self._get_ball_position_at_frame(ball_tracks, prev_holder_frame)
                                end_pos = self._get_ball_position_at_frame(ball_tracks, frame)
                                
                                if start_pos is not None and end_pos is not None:
                                    # Check if pass enters final third
                                    start_in_final_third = self._is_in_final_third(start_pos[0], prev_team)
                                    end_in_final_third = self._is_in_final_third(end_pos[0], prev_team)
                                    
                                    # Final third pass: starts outside, ends inside
                                    if not start_in_final_third and end_in_final_third:
                                        final_third_passes[frame] = prev_team
                                        
                                        pass_details.append({
                                            'frame': frame,
                                            'passer_id': prev_holder,
                                            'receiver_id': current_holder,
                                            'team_id': prev_team,
                                            'start_frame': prev_holder_frame,
                                            'end_frame': frame,
                                            'start_pos': start_pos.tolist(),
                                            'end_pos': end_pos.tolist()
                                        })
                
                # Update tracking
                prev_holder = current_holder
                prev_holder_frame = frame
        
        return final_third_passes, {'passes': pass_details}
    
    def get_final_third_statistics(
        self, 
        final_third_passes: List[int], 
        player_assignment: List[Dict[int, int]],
        pass_details: Dict
    ) -> Dict:
        """
        Calculate statistics for final third passes.
        
        Args:
            final_third_passes: List indicating final third passes per frame
            player_assignment: Team assignments per frame
            pass_details: Detailed pass information from detect_final_third_passes
            
        Returns:
            Dictionary with statistics per team and per player
        """
        stats = {
            'total_final_third_passes': 0,
            'by_team': {1: 0, 2: 0},
            'by_player': {}
        }
        
        # Count passes by team
        for frame_idx, team_id in enumerate(final_third_passes):
            if team_id != -1:
                stats['total_final_third_passes'] += 1
                stats['by_team'][team_id] += 1
        
        # Count passes by player
        if 'passes' in pass_details:
            for pass_info in pass_details['passes']:
                passer_id = pass_info['passer_id']
                if passer_id not in stats['by_player']:
                    stats['by_player'][passer_id] = {
                        'count': 0,
                        'team_id': pass_info['team_id']
                    }
                stats['by_player'][passer_id]['count'] += 1
        
        return stats
