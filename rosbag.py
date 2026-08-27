"""
Per-frame trunk detection on the live LiDAR stream.

Pipeline per incoming scan (no submap, no accumulation of points):
  1. look up GLIM's LiDAR pose for this scan's stamp
  2. pts_map = R @ pts_lidar + t              (whole frame, every frame)
  3. fold pts_map into the incremental ground grid (min-Z per XY cell)
  4. detect_trunks() on this frame alone, height-normalized by the ground grid
  5. match the (x, y, r) detections -- already world-frame -- into the tree registry

Pose source: /glim_ros/lidar_pose (geometry_msgs/PoseStamped).
  header.frame_id = "map", header.stamp = the scan stamp, pose = T_map_lidar
  (loop-closure corrected). We subscribe to the pose topic rather than doing a
  tf2 lookup on purpose -- see NOTE at the bottom of this file.
"""

import bisect
import math
from collections import defaultdict, deque

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, DurabilityPolicy
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import PointCloud2, Image, Imu
from tf2_msgs.msg import TFMessage
import sensor_msgs_py.point_cloud2 as pc2
from scipy import ndimage
from dendromatics.sections import fit_circle


# --------------------------------------------------------------------------- #
# tunables -- the slab/voxel/CCL numbers are the same ones used on the static
# GLIM map in voxelization.py, so per-frame results stay comparable.
# --------------------------------------------------------------------------- #
class Cfg:
    # detection (slab/voxel numbers match voxelization.py)
    voxel = 0.10             # m, XY raster + voxel dedup resolution
    z_min = 0.40             # m above ground, slab bottom
    z_max = 1.37             # m above ground, slab top
    n_min = 6                # min real returns in a component to fit a circle.
                             # 3 (as on the static map) leaves the fit
                             # unconstrained on a single sparse frame.
    group_dilate = 2         # cells. One frame puts only a handful of returns
                             # on a trunk, spread over ~3 cells, so a 3x3 close
                             # does not connect them. Dilate by this radius to
                             # GROUP returns, then fit on the undilated ones.
    resid_frac = 0.40        # reject a fit whose radial scatter exceeds this
                             # fraction of the fitted radius (scale-free)

    # ground grid
    ground_cell = 0.50       # m, min-Z cell size
    ground_min_hits = 3      # a cell is only trusted once this many points land in it
    ground_fill_rings = 2    # search this many ground cells outward when a cell is empty

    # per-frame detection range. Two things fall off with range: the ground
    # grid thins out (so height normalization stops being trustworthy) and the
    # slab returns per trunk drop below what a circle fit needs. Measured on
    # the recorded bag, association rate is clearly better at 10 m than 15 m.
    # Set to None to rasterize the whole frame instead.
    det_radius = 10.0        # m, horizontal, from the sensor origin

    # The sim only models the Mid-360's 0.1 m blind zone, so the robot's own
    # body and legs are returned and land squarely in the slab band -- on the
    # recorded bag a quarter of all slab points were inside 0.6 m, and they
    # fitted into a persistent phantom "trunk" at the robot's own position.
    self_radius = 0.8        # m, horizontal, drop returns closer than this

    # detection plausibility
    r_min = 0.03             # m, reject sub-3cm "trunks"
    r_max = 0.60             # m, reject blobs too fat to be a trunk

    # registry
    # Association gate, recomputed as detections come in:
    #     d_match = max(d_match_floor, largest trunk radius seen so far)
    # One frame sees only the near arc of a trunk, so the fitted centre can sit
    # up to ~r off the true axis -- the gate has to be at least as wide as the
    # fattest trunk actually present, not a fixed guess. r_max above caps what
    # can enter the registry at all, so the gate stays in
    # [d_match_floor, r_max] = [0.50, 0.60] m.
    d_match_floor = 5 * voxel   # m, 0.50: five raster cells
    reg_cell = 5.0           # m, spatial-hash cell so matching only looks at
                             # nearby entries, not the whole registry

    pose_max_dt = 0.05       # s, refuse to use a pose further than this from
                             # the scan stamp

    # GLIM publishes the pose for scan N about one scan period AFTER the scan
    # itself, so a scan has to wait for its pose. Hold at most this many.
    max_pending = 20         # scans (~2 s at 9.4 Hz)

    # logging. The per-frame detections and the local->global merge decisions
    # are the point of this script; the raw sensor stream is not.
    log_sensor_msgs = False  # per-message CAMERA / IMU / TF_STATIC lines
    quiet_every = 50         # frames with no detections: one line every N


def stamp_to_sec(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


# --------------------------------------------------------------------------- #
# pose buffer
# --------------------------------------------------------------------------- #
class PoseBuffer:
    """Time-ordered ring of GLIM LiDAR poses, with interpolated lookup.

    GLIM re-publishes corrected poses for stamps we may already hold, so an
    insert replaces any existing entry at the same stamp.
    """

    def __init__(self, maxlen=2000):
        self.maxlen = maxlen
        self.stamps = []        # sorted list of float seconds
        self.pos = []           # list of (3,) float64
        self.quat = []          # list of (4,) float64, [w, x, y, z]

    def add(self, stamp_sec, pos, quat):
        i = bisect.bisect_left(self.stamps, stamp_sec)
        if i < len(self.stamps) and abs(self.stamps[i] - stamp_sec) < 1e-9:
            self.pos[i], self.quat[i] = pos, quat      # corrected re-publish
            return
        self.stamps.insert(i, stamp_sec)
        self.pos.insert(i, pos)
        self.quat.insert(i, quat)
        if len(self.stamps) > self.maxlen:
            del self.stamps[0], self.pos[0], self.quat[0]

    def lookup(self, stamp_sec, max_dt):
        """Return (R, t) for stamp_sec, or None if we can't cover it."""
        if not self.stamps:
            return None
        i = bisect.bisect_left(self.stamps, stamp_sec)

        # exact hit (the common case: GLIM stamps its pose with the scan stamp)
        if i < len(self.stamps) and abs(self.stamps[i] - stamp_sec) < 1e-9:
            return quat_to_R(self.quat[i]), self.pos[i]

        if i == 0 or i == len(self.stamps):
            # outside the buffer -- only accept the nearest end if it's close
            j = 0 if i == 0 else len(self.stamps) - 1
            if abs(self.stamps[j] - stamp_sec) > max_dt:
                return None
            return quat_to_R(self.quat[j]), self.pos[j]

        t0, t1 = self.stamps[i - 1], self.stamps[i]
        if (stamp_sec - t0) > max_dt and (t1 - stamp_sec) > max_dt:
            return None
        a = (stamp_sec - t0) / (t1 - t0)
        pos = (1.0 - a) * self.pos[i - 1] + a * self.pos[i]
        quat = slerp(self.quat[i - 1], self.quat[i], a)
        return quat_to_R(quat), pos


def quat_to_R(q):
    """[w, x, y, z] -> 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def slerp(q0, q1, a):
    """Shortest-arc slerp between [w,x,y,z] quaternions."""
    d = float(np.dot(q0, q1))
    if d < 0.0:
        q1, d = -q1, -d
    if d > 0.9995:                                  # nearly parallel -> nlerp
        q = (1.0 - a) * q0 + a * q1
        return q / np.linalg.norm(q)
    th0 = math.acos(d)
    th = th0 * a
    s0 = math.sin(th0 - th) / math.sin(th0)
    s1 = math.sin(th) / math.sin(th0)
    return s0 * q0 + s1 * q1


# --------------------------------------------------------------------------- #
# incremental ground grid: min-Z per XY cell, in world (map) frame
# --------------------------------------------------------------------------- #
class GroundGrid:
    """Growable dense min-Z raster. Replaces the DTM used on the static map.

    Kept global (not scoped to a submap) because min-Z per cell is monotone and
    costs one np.minimum.at per frame -- there is nothing to bound.
    """

    def __init__(self, cell, min_hits, fill_rings):
        self.cell = cell
        self.min_hits = min_hits
        self.fill_rings = fill_rings
        self.origin = None                 # (i0, j0) cell index of array [0,0]
        self.min_z = None                  # float32, +inf where unseen
        self.count = None                  # uint32 hit count per cell

    def _cell_idx(self, xy):
        return np.floor(xy / self.cell).astype(np.int64)

    def _ensure(self, i_lo, i_hi, j_lo, j_hi):
        """Grow the array so [i_lo, i_hi] x [j_lo, j_hi] cell indices fit."""
        if self.origin is None:
            pad = 64
            self.origin = (i_lo - pad, j_lo - pad)
            shape = (i_hi - i_lo + 1 + 2 * pad, j_hi - j_lo + 1 + 2 * pad)
            self.min_z = np.full(shape, np.inf, dtype=np.float32)
            self.count = np.zeros(shape, dtype=np.uint32)
            return

        i0, j0 = self.origin
        h, w = self.min_z.shape
        gi_lo = min(0, i_lo - i0)
        gj_lo = min(0, j_lo - j0)
        gi_hi = max(h, i_hi - i0 + 1)
        gj_hi = max(w, j_hi - j0 + 1)
        if gi_lo == 0 and gj_lo == 0 and gi_hi == h and gj_hi == w:
            return

        pad = 64
        gi_lo -= pad if gi_lo < 0 else 0
        gj_lo -= pad if gj_lo < 0 else 0
        gi_hi += pad if gi_hi > h else 0
        gj_hi += pad if gj_hi > w else 0

        new_z = np.full((gi_hi - gi_lo, gj_hi - gj_lo), np.inf, dtype=np.float32)
        new_c = np.zeros(new_z.shape, dtype=np.uint32)
        new_z[-gi_lo:-gi_lo + h, -gj_lo:-gj_lo + w] = self.min_z
        new_c[-gi_lo:-gi_lo + h, -gj_lo:-gj_lo + w] = self.count
        self.min_z, self.count = new_z, new_c
        self.origin = (i0 + gi_lo, j0 + gj_lo)

    def update(self, pts_map):
        idx = self._cell_idx(pts_map[:, :2])
        self._ensure(idx[:, 0].min(), idx[:, 0].max(),
                     idx[:, 1].min(), idx[:, 1].max())
        i0, j0 = self.origin
        r = idx[:, 0] - i0
        c = idx[:, 1] - j0
        np.minimum.at(self.min_z, (r, c), pts_map[:, 2].astype(np.float32))
        np.add.at(self.count, (r, c), 1)

    def trusted(self):
        """Boolean mask of cells with enough hits to believe their min-Z."""
        return self.count >= self.min_hits

    def query(self, xy):
        """Ground Z under each XY. Returns (z, valid) with valid=False where no
        trusted cell was found within fill_rings."""
        n = len(xy)
        z = np.full(n, np.nan, dtype=np.float32)
        valid = np.zeros(n, dtype=bool)
        if self.origin is None:
            return z, valid

        i0, j0 = self.origin
        h, w = self.min_z.shape
        idx = self._cell_idx(xy)
        r = idx[:, 0] - i0
        c = idx[:, 1] - j0

        trusted = self.trusted()
        # rings 0..fill_rings: take the min over a growing square neighbourhood
        for ring in range(self.fill_rings + 1):
            todo = ~valid
            if not todo.any():
                break
            best = np.full(todo.sum(), np.inf, dtype=np.float32)
            rr, cc = r[todo], c[todo]
            for di in range(-ring, ring + 1):
                for dj in range(-ring, ring + 1):
                    if ring > 0 and abs(di) != ring and abs(dj) != ring:
                        continue          # interior already covered by prev ring
                    ri, cj = rr + di, cc + dj
                    ok = (ri >= 0) & (ri < h) & (cj >= 0) & (cj < w)
                    if not ok.any():
                        continue
                    v = np.full(len(ri), np.inf, dtype=np.float32)
                    v[ok] = np.where(trusted[ri[ok], cj[ok]],
                                     self.min_z[ri[ok], cj[ok]], np.inf)
                    best = np.minimum(best, v)
            got = np.isfinite(best)
            sel = np.flatnonzero(todo)[got]
            z[sel] = best[got]
            valid[sel] = True
        return z, valid


# --------------------------------------------------------------------------- #
# tree registry: spatial-hash so matching touches only nearby entries
# --------------------------------------------------------------------------- #
class TreeRegistry:
    def __init__(self, d_match_floor, cell):
        self.d_match_floor = d_match_floor
        self.r_seen_max = 0.0    # largest radius any scan has produced so far
        self.cell = cell
        self.xy = []             # list of np.array([x, y])
        self.r = []              # running mean radius
        self.n_obs = []
        self.last_frame = []
        self._hash = defaultdict(list)   # (ci, cj) -> [entry ids]

    @property
    def d_match(self):
        """Gate width: the floor, or the fattest trunk seen so far if wider."""
        return max(self.d_match_floor, self.r_seen_max)

    def _key(self, x, y):
        return (int(math.floor(x / self.cell)), int(math.floor(y / self.cell)))

    def _candidates(self, x, y):
        ci, cj = self._key(x, y)
        out = []
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                out.extend(self._hash.get((ci + di, cj + dj), ()))
        return out

    def match(self, dets, frame_idx):
        """Merge this frame's detections into the global list.

        dets: (N, 3) world-frame [x, y, r].

        For each detection, compare against the global list -- only the entries
        in the neighbouring hash cells, since anything outside them is further
        than d_match by construction. If the nearest entry is within d_match the
        detection IS that tree: fold it into the entry's running mean and add no
        new row. Otherwise it becomes a new row. The gate is self.d_match, which
        widens as fatter trunks turn up -- see Cfg.d_match_floor.

        Returns one decision per detection:
            ('merge', tree_id, distance_to_that_tree)
            ('new',   tree_id, (x, y, r))
        """
        events = []
        for x, y, rad in dets:
            # this detection counts towards "seen so far" before it is gated
            self.r_seen_max = max(self.r_seen_max, float(rad))
            cand = self._candidates(x, y)
            best, best_d = -1, self.d_match
            for k in cand:
                d = math.hypot(self.xy[k][0] - x, self.xy[k][1] - y)
                if d < best_d:
                    best, best_d = k, d
            if best >= 0:
                n = self.n_obs[best]
                # running mean of position and radius
                self.xy[best] = (self.xy[best] * n + np.array([x, y])) / (n + 1)
                self.r[best] = (self.r[best] * n + rad) / (n + 1)
                self.n_obs[best] = n + 1
                self.last_frame[best] = frame_idx
                events.append(('merge', best, best_d))
            else:
                k = len(self.xy)
                self.xy.append(np.array([x, y], dtype=np.float64))
                self.r.append(float(rad))
                self.n_obs.append(1)
                self.last_frame.append(frame_idx)
                self._hash[self._key(x, y)].append(k)
                events.append(('new', k, (float(x), float(y), float(rad))))
        return events

    def confirmed(self, min_obs=2):
        return np.array([[self.xy[k][0], self.xy[k][1], self.r[k], self.n_obs[k]]
                         for k in range(len(self.xy))
                         if self.n_obs[k] >= min_obs], dtype=np.float64)

    def __len__(self):
        return len(self.xy)


# --------------------------------------------------------------------------- #
# per-frame detection -- same stages as voxelization.py, DTM swapped for the
# incremental ground grid
# --------------------------------------------------------------------------- #
def detect_trunks(pts_map, sensor_xy, ground, cfg=Cfg):
    """pts_map: (N, 3) world-frame points from ONE scan.
    Returns (M, 3) world-frame [x, y, radius]."""
    empty = np.zeros((0, 3))
    if len(pts_map) == 0:
        return empty

    pts = pts_map
    d2 = ((pts[:, :2] - sensor_xy) ** 2).sum(axis=1)
    keep = d2 >= cfg.self_radius ** 2
    if cfg.det_radius is not None:
        keep &= d2 <= cfg.det_radius ** 2
    pts = pts[keep]
    if len(pts) == 0:
        return empty

    # --- height normalization against the incremental ground grid ---------
    gz, gok = ground.query(pts[:, :2])
    pts = pts[gok]
    if len(pts) == 0:
        return empty
    z0 = pts[:, 2] - gz[gok]

    # --- slab ------------------------------------------------------------
    slab = pts[(z0 >= cfg.z_min) & (z0 <= cfg.z_max)]
    if len(slab) < cfg.n_min:
        return empty

    # --- voxelize + XY projection in one step -----------------------------
    # voxel dedup and the 2D raster share the same resolution, so going
    # straight to unique XY cells is equivalent to building the voxel grid and
    # then projecting it (as voxelization.py does), minus the Open3D objects.
    ij = np.floor(slab[:, :2] / cfg.voxel).astype(np.int64)
    i_min, j_min = ij.min(axis=0)
    i_max, j_max = ij.max(axis=0)
    occ = np.zeros((i_max - i_min + 1, j_max - j_min + 1), dtype=bool)
    occ[ij[:, 0] - i_min, ij[:, 1] - j_min] = True

    # --- group + 8-connected labelling ------------------------------------
    # Dilate to decide WHICH returns belong to the same trunk, then throw the
    # dilation away and keep the labels only on real returns, so the circle is
    # fitted to measured points and not to padding.
    d = cfg.group_dilate
    if d > 0:
        yy, xx = np.mgrid[-d:d + 1, -d:d + 1]
        se = (yy ** 2 + xx ** 2) <= d ** 2 + 1e-9
        grouped = ndimage.binary_dilation(occ, structure=se)
    else:
        grouped = ndimage.binary_closing(occ, structure=np.ones((3, 3)))
    labels, n_comp = ndimage.label(grouped, structure=np.ones((3, 3), dtype=int))
    if n_comp == 0:
        return empty
    labels = np.where(occ, labels, 0)

    # --- circle fit per component ----------------------------------------
    # A single scan sees only the near arc of each trunk, so the component
    # pixels form an arc, not a filled disk -- which is exactly the model
    # fit_circle assumes.
    dets = []
    for label_id, sl in enumerate(ndimage.find_objects(labels), start=1):
        if sl is None:
            continue
        sub = labels[sl] == label_id
        if int(sub.sum()) < cfg.n_min:
            continue
        li, lj = np.nonzero(sub)
        gi = li + sl[0].start
        gj = lj + sl[1].start
        # fit in cell units, matching voxelization.py's (xs=col, ys=row)
        center, radius_px = fit_circle(gj.astype(float), gi.astype(float))
        radius = radius_px * cfg.voxel
        if not (cfg.r_min <= radius <= cfg.r_max):
            continue
        # scale-free goodness of fit: radial scatter vs fitted radius
        rr = np.hypot(gj - center[0], gi - center[1])
        if radius_px < 1e-6 or rr.std() / radius_px > cfg.resid_frac:
            continue
        # cell units -> world metres (+0.5 for cell centre)
        x = (center[1] + i_min + 0.5) * cfg.voxel   # center[1] is the row -> i -> x
        y = (center[0] + j_min + 0.5) * cfg.voxel   # center[0] is the col -> j -> y
        dets.append((x, y, radius))

    return np.array(dets, dtype=np.float64) if dets else empty


# --------------------------------------------------------------------------- #
# node
# --------------------------------------------------------------------------- #
class SensorStreamReader(Node):
    def __init__(self):
        super().__init__('sensor_stream_reader')
        self.set_parameters([Parameter('use_sim_time', Parameter.Type.BOOL, True)])

        self.cfg = Cfg
        self.poses = PoseBuffer()
        self.ground = GroundGrid(Cfg.ground_cell, Cfg.ground_min_hits,
                                Cfg.ground_fill_rings)
        self.registry = TreeRegistry(Cfg.d_match_floor, Cfg.reg_cell)
        self.n_skipped_no_pose = 0
        self.n_pose_msgs = 0
        self.pending = deque()   # scans held until their GLIM pose arrives

        self.lidar_count = 0
        self.sub_lidar = self.create_subscription(
            PointCloud2, '/unitree_go2/lidar/point_cloud', self.lidar_callback, 10)

        # GLIM's LiDAR pose in the map frame, loop-closure corrected.
        self.sub_pose = self.create_subscription(
            PoseStamped, '/glim_ros/lidar_pose', self.pose_callback, 100)
        self.sub_pose_corr = self.create_subscription(
            PoseStamped, '/glim_ros/lidar_pose_corrected', self.pose_callback, 100)

        self.camera_count = 0
        self.sub_camera = self.create_subscription(
            Image, '/unitree_go2/front_cam/color_image', self.camera_callback, 10)
        self.imu_count = 0
        self.sub_imu = self.create_subscription(
            Imu, '/unitree_go2/imu', self.imu_callback, 10)
        self.tf_sub = self.create_subscription(
            TFMessage, '/tf', self.tf_callback, 50)
        static_qos = QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.tf_static_sub = self.create_subscription(
            TFMessage, '/tf_static', self.tf_static_callback, static_qos)

        self.get_logger().info("Subscribed to lidar, camera, imu, tf, tf_static, glim pose")

    # ------------------------------------------------------------------ #
    def pose_callback(self, msg):
        p = msg.pose.position
        q = msg.pose.orientation
        self.n_pose_msgs += 1
        if self.n_pose_msgs == 1:
            self.get_logger().info(
                f"first GLIM pose received: stamp="
                f"{stamp_to_sec(msg.header.stamp):.3f} frame={msg.header.frame_id}")
        self.poses.add(
            stamp_to_sec(msg.header.stamp),
            np.array([p.x, p.y, p.z], dtype=np.float64),
            np.array([q.w, q.x, q.y, q.z], dtype=np.float64),
        )
        self._drain()

    def lidar_callback(self, msg):
        """Queue the scan; _drain() processes it once its pose exists.

        GLIM publishes the pose for scan N roughly one scan period AFTER the
        scan itself goes out, so looking the pose up here -- the instant the
        cloud arrives -- always asks for a stamp GLIM has not reached yet, and
        the newest pose on hand is one frame stale. Waiting costs one frame of
        latency and nothing else; using the stale pose would cost v * 0.106 s
        of position error on every single frame.
        """
        self.lidar_count += 1
        pts, _ = read_xyzt(msg)
        if pts is None or len(pts) == 0:
            return
        self.pending.append(
            (stamp_to_sec(msg.header.stamp), pts, self.lidar_count))
        self._drain()

    def _drain(self):
        """Process every queued scan whose pose has landed, oldest first."""
        while self.pending:
            stamp, pts, frame_idx = self.pending[0]
            pose = self.poses.lookup(stamp, self.cfg.pose_max_dt)
            if pose is None:
                newest = self.poses.stamps[-1] if self.poses.stamps else None
                # Drop it only once GLIM has published PAST this scan (so the
                # pose is genuinely missing, not merely late), or the queue has
                # outgrown its cap. Otherwise keep waiting.
                overtaken = newest is not None and newest > stamp + self.cfg.pose_max_dt
                if overtaken or len(self.pending) > self.cfg.max_pending:
                    self.pending.popleft()
                    self.n_skipped_no_pose += 1
                    if self.n_skipped_no_pose % 20 == 1:
                        self._warn_no_pose(stamp, frame_idx)
                    continue
                return
            self.pending.popleft()
            self._process(stamp, pts, frame_idx, pose)

    def _process(self, stamp, pts, frame_idx, pose):
        R, t = pose

        # 2. into world frame -- whole frame, every frame.
        #    Single rigid transform, NOT deskewed: the simulator casts all rays
        #    of a scan from one frozen sensor pose, so the cloud carries no
        #    intra-frame motion for a per-point pose to undo. See NOTE below.
        pts_map = pts @ R.T + t

        # 3. incremental ground grid
        self.ground.update(pts_map)

        # 4. detect on this frame alone
        dets = detect_trunks(pts_map, t[:2], self.ground, self.cfg)

        # 5. local -> global merge. Detections are already world-frame, so a
        #    detection is "the same tree" purely on XY proximity to an entry.
        events = self.registry.match(dets, frame_idx)

        if not events:
            if frame_idx % self.cfg.quiet_every == 0:
                self.get_logger().info(
                    f"frame {frame_idx}: 0 centroids | "
                    f"trees={len(self.registry)}")
            return

        merged = [(k, d) for kind, k, d in events if kind == 'merge']
        new = [(k, v) for kind, k, v in events if kind == 'new']
        parts = [f"frame {frame_idx}: {len(events)} centroids"]
        if merged:
            parts.append("same-tree " + ", ".join(
                f"#{k} (d={d:.2f}m)" for k, d in merged))
        if new:
            parts.append("NEW " + ", ".join(
                f"#{k} (x={v[0]:.2f} y={v[1]:.2f} r={v[2]:.2f})" for k, v in new))
        parts.append(f"trees={len(self.registry)} "
                     f"(gate={self.registry.d_match:.2f}m)")
        self.get_logger().info(" | ".join(parts))

    # ------------------------------------------------------------------ #
    def _warn_no_pose(self, stamp, frame_idx):
        """Say WHICH of the two failure modes this is: no poses arriving at
        all, or poses arriving whose stamps don't cover this scan."""
        if self.n_pose_msgs == 0:
            self.get_logger().warn(
                f"frame {frame_idx}: no GLIM pose for stamp={stamp:.3f} "
                f"-- 0 messages on /glim_ros/lidar_pose[_corrected]. GLIM is "
                f"not publishing: check `ros2 topic hz /glim_ros/lidar_pose`, "
                f"that glim_rosnode is up with librviz_viewer.so loaded, and "
                f"that it was started BEFORE the bag. "
                f"(skipped {self.n_skipped_no_pose})")
            return
        lo, hi = self.poses.stamps[0], self.poses.stamps[-1]
        i = bisect.bisect_left(self.poses.stamps, stamp)
        window = self.poses.stamps[max(0, i - 1):i + 1]
        near = min((abs(v - stamp) for v in window), default=float('inf'))
        self.get_logger().warn(
            f"frame {frame_idx}: no GLIM pose for stamp={stamp:.3f} -- "
            f"{self.n_pose_msgs} poses received, buffer spans "
            f"[{lo:.3f}..{hi:.3f}], nearest dt={near:.3f}s > pose_max_dt="
            f"{self.cfg.pose_max_dt}s (skipped {self.n_skipped_no_pose})")

    # ------------------------------------------------------------------ #
    def camera_callback(self, msg):
        self.camera_count += 1
        if not self.cfg.log_sensor_msgs:
            return
        self.get_logger().info(
            f"[CAMERA] frame {self.camera_count}: {msg.width}x{msg.height}, "
            f"encoding={msg.encoding}, stamp={msg.header.stamp.sec}.{msg.header.stamp.nanosec}")

    def imu_callback(self, msg):
        self.imu_count += 1
        if self.cfg.log_sensor_msgs and self.imu_count % 100 == 0:
            self.get_logger().info(
                f"[IMU] msg {self.imu_count}: "
                f"accel=({msg.linear_acceleration.x:.2f}, {msg.linear_acceleration.y:.2f}, {msg.linear_acceleration.z:.2f}), "
                f"stamp={msg.header.stamp.sec}.{msg.header.stamp.nanosec}")

    def tf_callback(self, msg):
        pass  # GLIM's pose comes from the pose topic, not TF -- see NOTE below

    def tf_static_callback(self, msg):
        if not self.cfg.log_sensor_msgs:
            return
        for transform in msg.transforms:
            self.get_logger().info(
                f"[TF_STATIC] {transform.header.frame_id} -> {transform.child_frame_id}")

    def save_registry(self, path="tree_registry.npy"):
        arr = self.registry.confirmed(min_obs=1)
        np.save(path, arr)
        # print, not get_logger: on Ctrl-C the rclpy context is already down
        # and rosout publishing fails.
        print(f"Saved {len(arr)} trees to {path} "
              f"({self.lidar_count} lidar frames, "
              f"{self.n_skipped_no_pose} skipped for want of a pose)")


def read_xyzt(msg):
    """(N,3) float64 xyz in the LiDAR frame + (N,) int64 per-point ns offsets."""
    arr = pc2.read_points(msg, field_names=("x", "y", "z", "t"), skip_nans=True)
    if arr.dtype.names is None:                       # older tuple-list API
        arr = np.asarray(list(arr), dtype=np.float64)
        if arr.size == 0:
            return None, None
        return arr[:, :3], arr[:, 3].astype(np.int64)
    if arr.shape[0] == 0:
        return None, None
    pts = np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(np.float64)
    return pts, arr["t"].astype(np.int64)


def main():
    rclpy.init()
    node = SensorStreamReader()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.save_registry()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()


# --------------------------------------------------------------------------- #
# NOTE 1 -- per-point timestamps are present but synthetic
#
# The cloud does carry a per-point time: field 't', UINT32 at byte offset 12,
# nanoseconds from header.stamp. Verified on the recorded bag: monotone,
# 0 .. 99,995,000 ns, unique per point, spanning the full 100 ms frame.
#
# But it is a label, not a measurement. In go2_sensors.py::get_pointcloud():
#     transform = self._get_sensor_transform()      # called ONCE per frame
#     origins   = np.tile(origin, (len(dirs_world), 1))
#     offsets_ns = point_indices / points_per_frame * frame_duration_ns
# All 20,000 rays are cast from one frozen sensor pose at one instant, and 't'
# is a linear ramp over the CSV scan-pattern row index. The geometry therefore
# contains zero intra-frame motion.
#
# So: one pose per frame is not an approximation here, it is exact. Deskewing
# with 't' would warp a cloud that was never warped, adding error of roughly
# v * 0.1 s (~5 cm at 0.5 m/s, worse while turning). Don't.
#
# NOTE 2 -- why the pose topic instead of a tf2 lookup
#
# The bag already contains map -> unitree_go2/base_link (Isaac's ground-truth
# pose, from the bridge) and a static base_link -> lidar_frame. Running GLIM on
# top adds map -> odom -> <base> -> lidar_frame, which gives lidar_frame two
# parents and makes a tf2 lookup of map -> lidar_frame ambiguous. Subscribing
# to /glim_ros/lidar_pose sidesteps that: it is T_map_lidar directly, stamped
# with the scan stamp, in the map frame.
#
# If you would rather use TF, replay the bag without its /tf
# (ros2 bag play ... --topics /unitree_go2/lidar/point_cloud /unitree_go2/imu
#  /tf_static) so only GLIM writes the map->...->lidar chain.
# --------------------------------------------------------------------------- #
