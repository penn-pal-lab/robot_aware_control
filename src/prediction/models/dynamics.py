import torch
from torch import cat


class DynamicsModel:
    def __init__(self, config):
        self._last_frame_skip = config.last_frame_skip
        self._skip = None
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        self._device = device
        self._stoch = config.stoch

    def reset(self, batch_size=None):
        """
        Call to reset any intermediate variables, like the ground truth skip cxn or
        batch size of hidden state for RNN
        """
        self._skip = None
        self.frame_predictor.hidden = self.frame_predictor.init_hidden(batch_size)
        if self._stoch:
            self.prior.hidden = self.prior.init_hidden(batch_size)
            self.prior.eval()
        self.decoder.eval()
        self.encoder.eval()
        self.robot_enc.eval()
        self.action_enc.eval()

    def load_model(self, model_path):
        d = self._device
        ckpt = torch.load(model_path, map_location=d)
        self.frame_predictor = ckpt["frame_predictor"].to(d)
        if self._stoch:
            # self.posterior = ckpt["posterior"]
            self.prior = ckpt["prior"].to(d)
        self.decoder = ckpt["decoder"].to(d)
        self.encoder = ckpt["encoder"].to(d)
        self.robot_enc = ckpt["robot_enc"].to(d)
        self.action_enc = ckpt["action_enc"].to(d)

    @torch.no_grad()
    def next_img(self, img, robot, action, store_skip=False):
        """
        Predicts the next img given current img, robot, and action
        F(s' | s, r, a)
        img is (N x |S|)
        action is (N x |A|)
        """
        h, skip = self.encoder(img)
        r = self.robot_enc(robot)
        a = self.action_enc(action)
        if store_skip:
            self._skip = skip
        if not self._last_frame_skip:
            skip = self._skip
        if self._stoch:
            z_t, _, _ = self.prior(cat([a, r, h], 1))
            h = self.frame_predictor(cat([a, r, h, z_t], 1))
        else:
            h = self.frame_predictor(cat([a, r, h], 1))
        next_img = self.decoder([h, skip])
        return next_img
