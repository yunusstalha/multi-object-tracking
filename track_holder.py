from track_object import TrackObject

class TrackHolder:
    
    def __init__(self):
        self.confirmed_tracks = []
        self.candidate_tracks = []
        self.old_tracks = []
        track_id_counter = 0
    
    def get_confirmed_tracks(self):
        """
        Returns the list of confirmed tracks.
        Parameters
        ----------
        None
        Returns
        -------
        list
            List of confirmed tracks.
        """
        return self.confirmed_tracks
    
    def get_candidate_tracks(self):
        """
        Returns the list of candidate tracks.
        Parameters
        ----------
        None
        Returns
        -------
        list
            List of candidate tracks.
        """
        return self.candidate_tracks
    def get_all_tracks(self):
        """
        Returns the list of all tracks.
        Parameters
        ----------
        None
        Returns
        -------
        list
            List of all tracks.
        """
        return self.confirmed_tracks + self.candidate_tracks
    
    def get_old_tracks(self):
        """
        Returns the list of old tracks.
        Parameters
        ----------
        None
        Returns
        -------
        list
            List of old tracks.
        """
        return self.old_tracks
   
    def kill_confirmed_tracks(self):
        """
        Removes a track from the list of confirmed tracks.
        Parameters
        ----------
        track : TrackObject
            Track to be removed.
        Returns
        -------
        None
        """
        M = 5
        N = 5

        for track in self.confirmed_tracks:
            mn_vector = track.get_streak_vector()
            for i in range(len(mn_vector)):
                # Get the last N samples or less if not enough data
                window = mn_vector[max(0, i-N+1):i+1]
                # Check if the number of 1's in the window is at least M
                if window.count(0) >= M:
                    self.confirmed_tracks.remove(track)
                    self.old_tracks.append(track)
                    break
    def kill_candidate_tracks(self):
        """
        Removes a track from the list of candidate tracks.
        Parameters
        ----------
        None
        Returns
        -------
        None
        """
        M = 3
        N = 3
        
        for track in self.candidate_tracks:
            mn_vector = track.get_streak_vector()
            for i in range(len(mn_vector)):
                # Get the last N samples or less if not enough data
                window = mn_vector[max(0, i-N+1):i+1]
                # Check if the number of 1's in the window is at least M
                if window.count(0) >= M:
                    self.candidate_tracks.remove(track)
                    break
                    # track.set_confirmed(False)
                    
    def add_candidate_track(self, track):
        """
        Adds a track to the list of candidate tracks.
        Parameters
        ----------
        track : TrackObject
            Track to be added.
        Returns
        -------
        None
        """
        self.candidate_tracks.append(track) 

    def confirm_candidate_tracks(self):
        """
        Confirms a track and adds it to the list of confirmed tracks.
        Parameters
        ----------
        None
        Returns
        -------
        None
        """
        M = 3
        N = 5

        for track in self.candidate_tracks:
            mn_vector = track.get_streak_vector()
            for i in range(len(mn_vector)):
                # Get the last N samples or less if not enough data
                window = mn_vector[max(0, i-N+1):i+1]
                # print(window)
                # Check if the number of 1's in the window is at least M
                if window.count(1) >= M:
                    self.confirmed_tracks.append(track)
                    self.candidate_tracks.remove(track)
                    # track.set_confirmed(True)
                    # track.set_track_id(self.track_id_counter)
                    # self.track_id_counter = self.track_id_counter + 1
                    break
