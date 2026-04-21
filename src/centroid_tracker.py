from collections import OrderedDict
import numpy as np
from scipy.spatial import distance as dist


class CentroidTracker:
    """Distance-based centroid tracker that counts unique objects over time."""

    def __init__(self, max_disappeared=20, max_distance=40):
        self.next_id = 0
        self.objects = OrderedDict()      # id -> centroid
        self.disappeared = OrderedDict()  # id -> consecutive frames missing
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.total_registered = 0

    def _register(self, centroid):
        self.objects[self.next_id] = centroid
        self.disappeared[self.next_id] = 0
        self.total_registered += 1
        self.next_id += 1

    def _deregister(self, obj_id):
        del self.objects[obj_id]
        del self.disappeared[obj_id]

    def update(self, centroids):
        """
        Update tracker with new centroids from current frame.
        Returns (active_objects dict, total unique count so far).
        """
        if len(centroids) == 0:
            for obj_id in list(self.disappeared):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self._deregister(obj_id)
            return self.objects, self.total_registered

        if len(self.objects) == 0:
            for c in centroids:
                self._register(c)
            return self.objects, self.total_registered

        obj_ids = list(self.objects.keys())
        obj_centroids = list(self.objects.values())

        D = dist.cdist(np.array(obj_centroids), np.array(centroids))
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows, used_cols = set(), set()
        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            if D[row, col] > self.max_distance:
                continue
            obj_id = obj_ids[row]
            self.objects[obj_id] = centroids[col]
            self.disappeared[obj_id] = 0
            used_rows.add(row)
            used_cols.add(col)

        for row in set(range(D.shape[0])) - used_rows:
            obj_id = obj_ids[row]
            self.disappeared[obj_id] += 1
            if self.disappeared[obj_id] > self.max_disappeared:
                self._deregister(obj_id)

        for col in set(range(D.shape[1])) - used_cols:
            self._register(centroids[col])

        return self.objects, self.total_registered
