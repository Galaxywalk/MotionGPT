import torch
import torch.nn as nn
from .base import BaseLosses


class CommitLoss(nn.Module):
    """
    Useless Wrapper
    """
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, commit, commit2, **kwargs):
        return commit


class GPTLosses(BaseLosses):
    
    def __init__(self, cfg, stage, num_joints, **kwargs):
        # Save parameters
        self.stage = stage
        recons_loss = cfg.LOSS.ABLATION.RECONS_LOSS
        self.speed_fps = float(cfg.LOSS.get("SPEED_FPS", 20.0))
        self.speed_domain = cfg.LOSS.get("SPEED_DOMAIN", "")
        self.traj_domain = cfg.LOSS.get("TRAJ_DOMAIN", "")

        # Define losses
        losses = []
        params = {}
        if stage == "vae":
            losses.append("recons_feature")
            params['recons_feature'] = cfg.LOSS.LAMBDA_FEATURE

            if cfg.LOSS.get("LAMBDA_ROOT", 0.0) != 0.0:
                losses.append("recons_root")
                params["recons_root"] = cfg.LOSS.LAMBDA_ROOT

            if cfg.LOSS.get("LAMBDA_ROOT_HEIGHT", 0.0) != 0.0:
                losses.append("recons_height")
                params["recons_height"] = cfg.LOSS.LAMBDA_ROOT_HEIGHT

            if cfg.LOSS.get("LAMBDA_TRAJ_FINAL", 0.0) != 0.0:
                losses.append("recons_trajfinal")
                params["recons_trajfinal"] = cfg.LOSS.LAMBDA_TRAJ_FINAL

            if cfg.LOSS.get("LAMBDA_TRAJ_PATH", 0.0) != 0.0:
                losses.append("recons_trajpath")
                params["recons_trajpath"] = cfg.LOSS.LAMBDA_TRAJ_PATH

            if cfg.LOSS.get("LAMBDA_TRAJ_STEP", 0.0) != 0.0:
                losses.append("recons_trajstep")
                params["recons_trajstep"] = cfg.LOSS.LAMBDA_TRAJ_STEP

            if cfg.LOSS.get("LAMBDA_SPEED_MEAN", 0.0) != 0.0:
                losses.append("recons_speedmean")
                params["recons_speedmean"] = cfg.LOSS.LAMBDA_SPEED_MEAN

            losses.append("recons_velocity")
            params['recons_velocity'] = cfg.LOSS.LAMBDA_VELOCITY

            losses.append("vq_commit")
            params['vq_commit'] = cfg.LOSS.LAMBDA_COMMIT
        elif stage in ["lm_pretrain", "lm_instruct"]:
            losses.append("gpt_loss")
            params['gpt_loss'] = cfg.LOSS.LAMBDA_CLS

        # Define loss functions & weights
        losses_func = {}
        for loss in losses:
            if loss in [
                    "recons_trajfinal",
                    "recons_trajpath",
                    "recons_trajstep",
                    "recons_speedmean",
            ]:
                losses_func[loss] = nn.L1Loss
            elif loss.split('_')[0] == 'recons':
                if recons_loss == "l1":
                    losses_func[loss] = nn.L1Loss
                elif recons_loss == "l2":
                    losses_func[loss] = nn.MSELoss
                elif recons_loss == "l1_smooth":
                    losses_func[loss] = nn.SmoothL1Loss
            elif loss.split('_')[1] in [
                    'commit', 'loss', 'gpt', 'm2t2m', 't2m2t'
            ]:
                losses_func[loss] = CommitLoss
            elif loss.split('_')[1] in ['cls', 'lm']:
                losses_func[loss] = nn.CrossEntropyLoss
            else:
                raise NotImplementedError(f"Loss {loss} not implemented.")

        super().__init__(cfg, losses, params, losses_func, num_joints,
                         **kwargs)

    def _select_domain_roots(self, rs_set, domain):
        root_ref = rs_set["root_ref"]
        root_rst = rs_set["root_rst"]
        if not domain or domain in ("all", "*"):
            return root_ref, root_rst

        domains = rs_set.get("domain")
        if domains is None:
            return None, None

        mask = torch.tensor(
            [item == domain for item in domains],
            device=root_ref.device,
            dtype=torch.bool,
        )
        if not bool(mask.any()):
            return None, None
        return root_ref[mask], root_rst[mask]

    def update(self, rs_set):
        '''Update the losses'''
        total: float = 0.0

        if self.stage in ["vae"]:
            total += self._update_loss("recons_feature", rs_set['m_rst'],
                                       rs_set['m_ref'])
            # total += self._update_loss("recons_joints", rs_set['joints_rst'], rs_set['joints_ref'])
            nfeats = rs_set['m_rst'].shape[-1]
            if "recons_root" in self._params:
                total += self._update_loss(
                    "recons_root",
                    rs_set["m_rst"][..., :3],
                    rs_set["m_ref"][..., :3],
                )
            if "recons_height" in self._params:
                total += self._update_loss(
                    "recons_height",
                    rs_set["m_rst"][..., 3:4],
                    rs_set["m_ref"][..., 3:4],
                )
            root_ref, root_rst = self._select_domain_roots(
                rs_set, self.traj_domain)
            if "recons_trajfinal" in self._params and root_ref is not None:
                ref_disp = (
                    root_ref[..., -1, [0, 2]] -
                    root_ref[..., 0, [0, 2]]
                )
                rst_disp = (
                    root_rst[..., -1, [0, 2]] -
                    root_rst[..., 0, [0, 2]]
                )
                total += self._update_loss("recons_trajfinal", rst_disp, ref_disp)
            if "recons_trajpath" in self._params and root_ref is not None:
                ref_steps = torch.linalg.norm(
                    root_ref[..., 1:, [0, 2]] -
                    root_ref[..., :-1, [0, 2]],
                    dim=-1,
                )
                rst_steps = torch.linalg.norm(
                    root_rst[..., 1:, [0, 2]] -
                    root_rst[..., :-1, [0, 2]],
                    dim=-1,
                )
                total += self._update_loss(
                    "recons_trajpath",
                    rst_steps.sum(dim=-1),
                    ref_steps.sum(dim=-1),
                )
            if "recons_trajstep" in self._params and root_ref is not None:
                ref_delta = (
                    root_ref[..., 1:, [0, 2]] -
                    root_ref[..., :-1, [0, 2]]
                )
                rst_delta = (
                    root_rst[..., 1:, [0, 2]] -
                    root_rst[..., :-1, [0, 2]]
                )
                total += self._update_loss("recons_trajstep", rst_delta,
                                           ref_delta)
            if "recons_speedmean" in self._params:
                root_ref, root_rst = self._select_domain_roots(
                    rs_set, self.speed_domain)
                if root_ref is not None and root_rst is not None:
                    ref_speed = torch.linalg.norm(
                        root_ref[..., 1:, [0, 2]] - root_ref[..., :-1, [0, 2]],
                        dim=-1,
                    ) * self.speed_fps
                    rst_speed = torch.linalg.norm(
                        root_rst[..., 1:, [0, 2]] - root_rst[..., :-1, [0, 2]],
                        dim=-1,
                    ) * self.speed_fps
                    total += self._update_loss(
                        "recons_speedmean",
                        rst_speed.mean().reshape(1),
                        ref_speed.mean().reshape(1),
                    )
            if nfeats in [263, 135 + 263]:
                if nfeats == 135 + 263:
                    vel_start = 135 + 4
                elif nfeats == 263:
                    vel_start = 4
                total += self._update_loss(
                    "recons_velocity",
                    rs_set['m_rst'][..., vel_start:(self.num_joints - 1) * 3 +
                                    vel_start],
                    rs_set['m_ref'][..., vel_start:(self.num_joints - 1) * 3 +
                                    vel_start])
            else:
                if self._params['recons_velocity'] != 0.0:
                    raise NotImplementedError(
                        "Velocity not implemented for nfeats = {})".format(nfeats))
            total += self._update_loss("vq_commit", rs_set['loss_commit'],
                                       rs_set['loss_commit'])

        if self.stage in ["lm_pretrain", "lm_instruct"]:
            total += self._update_loss("gpt_loss", rs_set['outputs'].loss,
                                       rs_set['outputs'].loss)

        # Update the total loss
        self.total += total.detach()
        self.count += 1

        return total
