import torch


def compute_force(nodes, f, skin_friction_lift=True):

    c_pressure_lift, c_pressure_drag = _compute_force_from_pressure(nodes, f[:, 0])

    if skin_friction_lift:
        c_friction_lift, c_friction_drag = _compute_force_from_friction(nodes, f[:, 1:])
        lift = c_pressure_lift + c_friction_lift
    else:
        c_friction_drag = _compute_drag_from_friction(nodes, f[:, 1:])
        lift = c_pressure_lift

    drag = c_pressure_drag + c_friction_drag

    return lift, drag


def _compute_force_from_pressure(nodes, p):

    p = p.reshape(-1)

    segs = torch.roll(nodes, shifts=-1, dims=0) - nodes
    dx = segs[:, 0]
    dy = segs[:, 1]

    nx_times_length = dy
    ny_times_length = -dx

    p_avg = 0.5 * (torch.roll(p, shifts=-1, dims=0) + p)

    fx = -p_avg * nx_times_length
    fy = -p_avg * ny_times_length

    drag = fx.sum()
    lift = fy.sum()

    return lift, drag


def _compute_force_from_friction(nodes, f):

    f = f.reshape(-1, 2)

    segs = torch.roll(nodes, shifts=-1, dims=0) - nodes
    ds = torch.linalg.norm(segs, dim=1)

    f_avg = 0.5 * (torch.roll(f, shifts=-1, dims=0) + f)

    drag = (f_avg[:, 0] * ds).sum()
    lift = (f_avg[:, 1] * ds).sum()

    return lift, drag


def _compute_drag_from_friction(nodes, f):

    f = f.reshape(-1, 1)

    segs = torch.roll(nodes, shifts=-1, dims=0) - nodes
    ds = torch.linalg.norm(segs, dim=1)

    f_avg = 0.5 * (torch.roll(f, shifts=-1, dims=0) + f)

    drag = (f_avg[:, 0] * ds).sum()

    return drag
