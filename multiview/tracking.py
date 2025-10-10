import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

def linear_assignment(cost_matrix):
    x, y = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))

def mpjpe(joints1, joints2):
    """
    joints1, joints2: [N_joints, 3]
    """
    return np.mean(np.linalg.norm(joints1 - joints2, axis=1)) * 1000

def mpjpe_batch(dets, trackers):
    """
    Computes MPJPE distance between all dets and trackers
    dets: (N_dets, N_joints, 3)
    trackers: (N_tracks, N_joints, 3)
    Returns: (N_dets, N_tracks) cost matrix
    """
    cost_matrix = np.zeros((len(dets), len(trackers)), dtype=np.float32)
    for d, det in enumerate(dets):
        for t, trk in enumerate(trackers):
            cost_matrix[d, t] = mpjpe(det, trk)
    return cost_matrix

class KalmanPoseTracker(object):
    count = 0
    def __init__(self, joints, payload=None):
        """
        Initialize tracker with joints: [N_joints, 3]
        and optional payload
        """
        N_joints = joints.shape[0]
        self.kf = KalmanFilter(dim_x=N_joints * 3, dim_z=N_joints * 3)

        self.kf.F = np.eye(N_joints * 3)
        self.kf.H = np.eye(N_joints * 3)
        self.kf.P *= 10.
        self.kf.R *= 1.
        self.kf.Q *= 0.01

        self.kf.x[:N_joints * 3, 0] = joints.reshape(-1)

        self.time_since_update = 0
        self.id = KalmanPoseTracker.count
        KalmanPoseTracker.count += 1
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

        self.payload = payload  # payload can be any dict or tensor

    def update(self, joints, payload=None):
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(joints.reshape(-1))

        if payload is not None:
            self.payload = payload  # Update payload if given

    def predict(self):
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        pred = self.kf.x[:].reshape(-1, 3)
        return pred

    def get_state(self):
        """
        Returns: (joints [N_joints, 3], payload)
        """
        return self.kf.x.reshape(-1, 3), self.payload

def associate_detections_to_trackers(detections, trackers, mpjpe_threshold=150.0):
    if len(trackers) == 0:
        return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,),dtype=int)

    cost_matrix = mpjpe_batch(detections, trackers)

    if min(cost_matrix.shape) > 0:
        matched_indices = linear_assignment(cost_matrix)
    else:
        matched_indices = np.empty((0,2))

    unmatched_detections = []
    unmatched_trackers = []
    matches = []

    for d, det in enumerate(detections):
        if d not in matched_indices[:,0]:
            unmatched_detections.append(d)
    for t, trk in enumerate(trackers):
        if t not in matched_indices[:,1]:
            unmatched_trackers.append(t)

    for m in matched_indices:
        if cost_matrix[m[0], m[1]] > mpjpe_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1,2))

    if len(matches) == 0:
        matches = np.empty((0,2),dtype=int)
    else:
        matches = np.concatenate(matches,axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

class Sort(object):
    def __init__(self, max_age=1, min_hits=3, mpjpe_threshold=300.0):
        """
        Args:
            max_age: maximum number of missed frames before a track is deleted
            min_hits: minimum number of hits to start outputting a track
            mpjpe_threshold: matching threshold in mm
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.mpjpe_threshold = mpjpe_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, joints_batch=np.empty((0, 17, 3)), payloads_batch=None):
        """
        Args:
            joints_batch: [N_humans, N_joints, 3] joints array
            payloads_batch: list of payloads [N_humans], each is a dict or tensor

        Returns:
            list of (joints_with_trackid: [N_joints, 4], payload) pairs
        """
        self.frame_count += 1
        ret = []

        if payloads_batch is None:
            payloads_batch = [None] * len(joints_batch)

        # Predict current trackers
        trks = []
        for t in self.trackers:
            pred = t.predict()
            trks.append(pred)
        trks = np.array(trks)

        # Match detections to trackers
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            joints_batch, trks, self.mpjpe_threshold)

        # Update matched trackers
        for m in matched:
            det_idx, trk_idx = m[0], m[1]
            self.trackers[trk_idx].update(joints_batch[det_idx], payloads_batch[det_idx])

        # Initialize new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanPoseTracker(joints_batch[i], payloads_batch[i])
            self.trackers.append(trk)

        # Prepare outputs
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            joints, payload = trk.get_state()
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                track_id = np.ones((joints.shape[0], 1)) * (trk.id + 1)  # positive id
                joints_with_id = np.concatenate((joints, track_id), axis=1)
                ret.append((joints_with_id, payload))
            i -= 1
            # Remove dead tracks
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)

        return ret
