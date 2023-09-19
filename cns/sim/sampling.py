import numpy as np
from scipy.spatial.transform import Rotation as R


###################### sample pose ######################

def sample_camera_pose(r_min, r_max, phi_min, phi_max, drz_max, dry_max, drx_max) -> np.ndarray:
    r = np.random.uniform(r_min, r_max)
    theta = np.random.uniform(-np.pi, np.pi)
    z = np.random.uniform(np.sin(phi_min/180*np.pi), np.sin(phi_max/180*np.pi))
    phi = np.arcsin(z)
    x = np.cos(phi) * np.cos(theta)
    y = np.cos(phi) * np.sin(theta)
    wct = np.array([x, y, z]) * r

    z_vec = -np.array([x, y, z])
    assert z_vec[-1] < 0
    y_vec = np.array([
        np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta),
        -np.cos(phi)
    ])
    x_vec = np.cross(y_vec, z_vec)
    wcR = np.stack([x_vec, y_vec, z_vec], axis=1)
    dRz = R.from_rotvec(np.array([0, 0, 1]) * np.random.uniform(-drz_max, drz_max) / 180*np.pi)
    dRy = R.from_rotvec(np.array([0, 1, 0]) * np.random.uniform(-dry_max, dry_max) / 180*np.pi)
    dRx = R.from_rotvec(np.array([1, 0, 0]) * np.random.uniform(-drx_max, drx_max) / 180*np.pi)
    wcR = wcR @ dRz.as_matrix() @ dRy.as_matrix() @ dRx.as_matrix()

    wcT = np.eye(4)
    wcT[:3, :3] = wcR
    wcT[:3, 3] = wct

    return wcT


###################### sample points ######################

def uniform_ball_sample(num, dim) -> np.ndarray:
    """Sample N points in a d-dimension unit ball randomly and evenly. 
    Use Muller / Marsaglia ('Normalized Gaussians') Method.

    Returns:
    - points: (num, dim)
    """
    u = np.random.normal(size=(num, dim))
    dev = np.linalg.norm(u, axis=-1, keepdims=True)
    radius = np.power(np.random.rand(num, 1), 1./dim)
    points = u * radius / dev
    return points


def uniform_ellipse_2d_sample(num, a, b) -> np.ndarray:
    u = np.random.random(num) / 4.0
    theta = np.arctan2(b*np.sin(2*np.pi*u), a*np.cos(2*np.pi*u))
    v = np.random.random(num)

    mask = (v >= 0.25) & (v < 0.5)
    theta[mask] = np.pi - theta[mask]
    mask = (v >= 0.5) & (v < 0.75)
    theta[mask] = np.pi + theta[mask]
    mask = v >= 0.75
    theta[mask] = -theta[mask]

    max_radius = a * b / np.sqrt((b*np.cos(theta))**2 + (a*np.sin(theta))**2)
    rand_radius = max_radius * np.sqrt(np.random.random(num))

    points = np.stack([rand_radius * np.cos(theta),
                       rand_radius * np.sin(theta)], axis=-1)
    return points


def gen_virtual_points_uniformed(num_pts, obs_r, obs_h) -> np.ndarray:
    """Generate 3d observation points randomly distributed in a cylinder.

    Arguments:
    - num_pts: number of points, should be larger than or equal to 3
    - obs_r: radius of observation space
    - obs_h: height of observation space

    Returns:
    - position: (num_pts, 3), sampled observation
    """
    num_pts = np.clip(num_pts, 3, None)
    while True:
        # uniformly distributed in a horizontal circle
        position = uniform_ball_sample(num_pts, 2) * obs_r
        # uniformly distributed vertically
        position = np.concatenate([position, (np.random.rand(num_pts, 1)-0.5)*obs_h], axis=-1)  # (N, 3)
        # overvall, uniformly distributed in a cyclinder with r=obs_r and h=obs_h

        if num_pts < 32:  # a reasonable large number
            # when number of points are small, points may lay in a line,
            # thus we need to reject the sample
            dist = np.linalg.norm(position[:, None, :] - position[None, :, :], axis=-1)  # (N, N)
            dist_index = np.argmax(dist, axis=-1)  # (N,)
            dist = dist[np.arange(num_pts), dist_index]  # (N,)
            # for each point, find the point with farthest distance

            # 对于每个点和与其最远点连线形成的直线，记为vec
            vec = position[dist_index] - position  # (N, 3)
            vec0 = np.repeat(vec[None, :, :], num_pts, axis=0).reshape((num_pts*num_pts, 3))
            vec1 = np.repeat(vec[:, None, :], num_pts, axis=1).reshape((num_pts*num_pts, 3))
            angle = np.arctan2(
                np.linalg.norm(np.cross(vec0, vec1), axis=-1),  # |a||b|sin(\theta)
                np.sum(vec0*vec1, axis=-1)  # |a||b|cos(\theta)
            ).reshape((num_pts, num_pts)) / np.pi * 180.  # (N, N)
            # 计算每组vec之间的夹角
            
            angle = np.abs(angle) % 180.
            angle = np.where(angle < 90, angle, 180-angle)
            angle_index = np.argmax(angle, axis=-1)  # (N,)
            angle = angle[np.arange(num_pts), angle_index]
            # 对于每个vec，找到与其夹角最大的另一个vec，并记录index与夹角

            mask = (dist > 0.5 * obs_r) & (dist[angle_index] > 0.5 * obs_r) & (angle > 40)
            # if points are two close or are almost in a line, reject this sampling
            if mask.sum() >= (num_pts+6)//3:
                break
        else:
            # if points are much enough, the probability for points lying in one line is small,
            # thus we could directly accept the sampling
            break
    return position


def gen_virtual_points_clustered(num_pts, num_clusters, obs_r, obs_h) -> np.ndarray:
    """Generate 3d observation points randomly distributed in a cylinder.

    Arguments:
    - num_pts: number of points, should be larger than or equal to 3
    - obs_r: radius of observation space
    - obs_h: height of observation space

    Returns:
    - position: (num_pts, 3), sampled observation
    """

    assert num_clusters <= num_pts

    # random assign points number to each cluster, 
    # but ensure each cluster contains at least one point
    num_points_per_cluster = np.ones(num_clusters, dtype=np.int32)
    add_indices = np.random.choice(num_clusters, size=num_pts-num_clusters, replace=True)
    np.add.at(num_points_per_cluster, add_indices, 1)

    all_points = []
    # cluster_centers = uniform_ball_sample(num_clusters, 2) * obs_r
    cluster_centers = gen_virtual_points_uniformed(num_clusters, obs_r, obs_h)
    for i in range(num_clusters):
        max_radius = obs_r / np.log2(num_clusters) * 1.5
        a = np.random.uniform(0.2 * max_radius, 0.8 * max_radius)
        b = max_radius - a

        points = uniform_ellipse_2d_sample(num_points_per_cluster[i], a, b)
        points = np.concatenate([points, (np.random.rand(len(points), 1)-0.5)*obs_h*0.2], axis=-1)  # (N, 3)
        # points = np.concatenate([points, np.zeros((len(points), 1))], axis=-1)  # (N, 3)

        drz_max = 180  # degree
        dry_max = drx_max = 45  # degree
        dRz = R.from_rotvec(np.array([0, 0, 1]) * np.random.uniform(-drz_max, drz_max) / 180*np.pi)
        dRy = R.from_rotvec(np.array([0, 1, 0]) * np.random.uniform(-dry_max, dry_max) / 180*np.pi)
        dRx = R.from_rotvec(np.array([1, 0, 0]) * np.random.uniform(-drx_max, drx_max) / 180*np.pi)

        dpose = dRz.as_matrix() @ dRy.as_matrix() @ dRx.as_matrix()
        points = points @ dpose.T
        points += cluster_centers[i]

        all_points.append(points)
    
    all_points = np.concatenate(all_points, axis=0)
    return all_points


def gen_virtual_points(num_pts, obs_r, obs_h) -> np.ndarray:
    num_pts = max(num_pts, 4)
    if num_pts < 20:
        return gen_virtual_points_uniformed(num_pts, obs_r, obs_h)
    else:
        num_uniformed_pts = int(round(num_pts * 0.2))
        num_clustered_pts = num_pts - num_uniformed_pts
        num_clusters = np.random.randint(3, int(np.log2(num_pts+1e-8)))

        # print("[INFO] num_clusters pick from [{}, {}] = {}"
        #       .format(3, int(np.sqrt(num_pts+1e-8)), num_clusters))

        uniformed_pts = gen_virtual_points_uniformed(num_uniformed_pts, obs_r, obs_h)
        clustered_pts = gen_virtual_points_clustered(num_clustered_pts, num_clusters, obs_r, obs_h)
        pts = np.concatenate([uniformed_pts, clustered_pts], axis=0)
        return pts


###################### add non-idealities ######################

class Perlin1d(object):
    def __init__(self, amp, freq):
        self.amp = amp
        self.freq = freq
        self.prev_floor = None
        self.offset = np.random.rand()
        self.grad = [np.random.uniform(-1, 1), np.random.uniform(-1, 1)]
    
    @classmethod
    def interp(cls, a0, a1, w):
        return a0 + (a1 - a0) * (6*w**5 - 15*w**4 + 10*w**3)
    
    def __call__(self, t):
        if np.isnan(t):
            # print("[WARN] NaN occurs in Perlin1d")
            self.prev_floor = None
            t = 0.0
        
        t = t * self.freq + self.offset
        s = int(np.floor(t))
        if s != self.prev_floor:
            self.prev_floor = s
            self.grad = [self.grad[1], np.random.uniform(-1, 1)]
        
        n0 = self.grad[0] * (t - s)
        n1 = self.grad[1] * (t - (s+1))
        n = self.interp(n0, n1, t-s)

        return n * self.amp


class SmoothRandomWalker(object):
    def __init__(self, lower, upper, freq, center=None) -> None:
        assert len(lower) == len(upper) == len(freq)
        if center is not None:
            assert len(lower) == len(center)
            self.center = np.array(center)
        else:
            self.center = (np.array(lower) + np.array(upper)) / 2

        num_dim = len(lower)
        self.perlins = [Perlin1d(upper[i] - lower[i], freq[i]) for i in range(num_dim)]
    
    def __call__(self, t):
        offset = np.array([perlin(t) for perlin in self.perlins])
        pos = offset + self.center
        return pos


class ObservationSampler(object):
    def __init__(self, points, tau_max=5):
        self.points = points[:, :2]  # assume [u, v] is the first two features

        freq = 1
        lower = np.min(points, axis=0)
        upper = np.max(points, axis=0)
        center = np.median(points, axis=0)
        center += (upper - lower) * np.random.uniform(-0.3, 0.3, len(lower))

        self.sigma = np.linalg.norm(upper - lower) * 0.15
        self.walkers = [SmoothRandomWalker(lower, upper, [freq] * len(lower), center)
                        for _ in range(3)]
        self.walker_centers = [walker(0) for walker in self.walkers]
    
        N = len(self.points)
        self.state = np.random.uniform(0, 1, size=N) < 0.5  # 1 means observed, 0 means missing
        self.last_time = np.zeros(N)  # last time transition occurs
        self.tau_total = np.random.uniform(0.5 * tau_max, tau_max, size=N)
        self.no_event_rval = np.random.uniform(0, 1, size=N)

        self.dpose_norm_accum = 0
        self.prev_wcT = None
    
    def gaussian(self, x, mu, sigma):
        return np.exp(-(x - mu)**2 / (2*sigma**2))
    
    def prob_distribution(self, xlim, ylim, image_hw):
        x_min, x_max = xlim
        y_min, y_max = ylim
        image_h, image_w = image_hw

        gx, gy = np.meshgrid(
            np.linspace(x_min, x_max, num=image_w, endpoint=True),
            np.linspace(y_min, y_max, num=image_h, endpoint=True),
            indexing="xy"
        )
        grid_coords = np.stack([gx, gy], axis=-1)  # (H, W, 2)
        observable_prob = np.zeros((image_h, image_w))
        for center_pos in self.walker_centers:
            dist = np.linalg.norm(grid_coords - center_pos, axis=-1)
            prob = self.gaussian(dist, 0, self.sigma)
            observable_prob = np.maximum(observable_prob, prob)
        return observable_prob
    
    def __call__(self, time, current_wcT, dist_scale: float):
        if self.prev_wcT is None:
            dpose_norm = 0
        else:
            dpose = np.linalg.inv(self.prev_wcT) @ current_wcT
            drot = R.from_matrix(dpose[:3, :3]).as_rotvec()
            dtrans = dpose[:3, 3]
            # TODO: use more reasonable strategy to calculate dpose_norm
            dpose_norm = np.linalg.norm(drot) / np.pi * 0.2 + \
                         np.linalg.norm(dtrans) / dist_scale * 0.7
        self.dpose_norm_accum += dpose_norm
        self.prev_wcT = current_wcT

        self.walker_centers = [walker(self.dpose_norm_accum) for walker in self.walkers]
        tau_to_missing_ratio = np.zeros(len(self.points))
        for center_pos in self.walker_centers:
            dist = np.linalg.norm(self.points - center_pos, axis=-1)
            prob = self.gaussian(dist, 0, self.sigma)
            tau_to_missing_ratio = np.maximum(tau_to_missing_ratio, prob)
            # higher prob means longer time for state transition
            # from observed state to missing state
        
        tau_to_missing = self.tau_total * tau_to_missing_ratio
        tau_to_observed = self.tau_total * (1 - tau_to_missing_ratio)

        tau = np.empty_like(self.tau_total)
        tau[self.state] = tau_to_missing[self.state]
        tau[~self.state] = tau_to_observed[~self.state]

        # expectation of -ln(x) where x~U(0,1) is 1
        # thus E(no_event_time) = no_event_time but incoporate some randomness
        no_event_time = tau * (- np.log(self.no_event_rval))
        trans_mask = (time - self.last_time) > no_event_time
        if trans_mask.any():
            self.state[trans_mask] = ~self.state[trans_mask]
            self.no_event_rval[trans_mask] = np.random.uniform(0, 1, size=trans_mask.sum())
        
        unobserved_index = np.nonzero(self.state == 0)[0]
        return unobserved_index, tau_to_missing_ratio


class ObservationMismatcher(object):
    def __init__(self, tau, ratio):
        self.tau = tau
        self.ratio = ratio
        self.accum_time = 0
        self.interval = None
        self.indices = None
    
    def reset(self):
        self.accum_time = 0
        self.interval = None
        self.indices = None
    
    def __call__(self, N, dt):
        if self.indices is None or (self.accum_time + dt > self.interval):
            self.accum_time = 0
            self.interval = -np.log(np.random.rand()) * self.tau
            self.indices = np.random.choice(N, int(N*self.ratio), replace=False)
        else:
            self.accum_time += dt
        return self.indices


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    def set_axes_equal(ax):
        """Set 3D plot axes to equal scale.

        Make axes of 3D plot have equal scale so that spheres appear as
        spheres and cubes as cubes.  Required since `ax.axis('equal')`
        and `ax.set_aspect('equal')` don't work on 3D.
        """
        limits = np.array([
            ax.get_xlim3d(),
            ax.get_ylim3d(),
            ax.get_zlim3d(),
        ])
        origin = np.mean(limits, axis=1)
        radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
        _set_axes_radius(ax, origin, radius)


    def _set_axes_radius(ax, origin, radius):
        x, y, z = origin
        ax.set_xlim3d([x - radius, x + radius])
        ax.set_ylim3d([y - radius, y + radius])
        ax.set_zlim3d([z - radius, z + radius])

    # points = uniform_ellipse_2d_sample(1000, a=5, b=1)
    # plt.figure()
    # plt.plot(points[:, 0], points[:, 1], ".")
    # plt.axis("equal")
    # plt.show()

    # points = gen_virtual_points_clustered(16, num_clusters=4, obs_r=0.2, obs_h=0.02)
    # points = gen_virtual_points(256, 0.2, 0.08)
    points = gen_virtual_points(256, 0.2, 0.1)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2])
    set_axes_equal(ax)
    plt.tight_layout()
    plt.show()

