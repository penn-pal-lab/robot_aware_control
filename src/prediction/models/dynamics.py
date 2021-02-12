import torch
import torch.nn as nn
from torch import cat
from src.prediction.models.base import MLPEncoder, init_weights
from src.prediction.models.lstm import LSTM, GaussianLSTM

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


class DeterministicModel(nn.Module):
    def __init__(self, config, input_dim=None):
        super().__init__()
        self._config = cf = config
        self._device = config.device
        if input_dim is None:
            input_dim = cf.action_enc_dim + cf.robot_enc_dim + cf.g_dim
        self._init_models(input_dim)

    def _init_models(self, input_dim):
        cf = self._config
        self.frame_predictor = frame_pred = LSTM(
            input_dim,
            cf.g_dim,
            cf.rnn_size,
            cf.predictor_rnn_layers,
            cf.batch_size,
        ).to(self._device)

        if cf.image_width == 64:
            from src.prediction.models.vgg_64 import Decoder, Encoder
        elif cf.image_width == 128:
            from src.prediction.models.vgg import Decoder, Encoder

        # RGB + mask channel
        self.encoder = enc = Encoder(
            cf.g_dim, cf.channels + 1, cf.multiview, cf.dropout
        ).to(self._device)
        self.decoder = dec = Decoder(cf.g_dim, cf.channels, cf.multiview).to(
            self._device
        )
        self.action_enc = ac = MLPEncoder(cf.action_dim, cf.action_enc_dim, 32).to(
            self._device
        )
        self.robot_enc = rob = MLPEncoder(cf.robot_dim, cf.robot_enc_dim, 32).to(
            self._device
        )
        self.all_models = []
        self.all_models.extend([frame_pred, enc, dec, ac, rob])
        for model in self.all_models:
            model.apply(init_weights)

    def init_hidden(self, batch_size=None):
        """
        Initialize the recurrent states by batch size
        """
        self.frame_predictor.hidden = self.frame_predictor.init_hidden(batch_size)

    def forward(self, image, mask, robot, action, skip=None):
        """Given current state, predict the next state

        Args:
            image ([Tensor]): batch of images
            mask ([Tensor]): batch of robot masks
            robot ([Tensor]): batch of robot states
            action ([Tensor]): batch of robot actions
            skip ([Tensor]): batch of skip connections

        Returns:
            [Tuple]: Next image, next latent, skip connection
        """
        h, curr_skip = self.encoder(cat([image, mask], dim=1))
        if skip is None:
            skip = curr_skip

        r = self.robot_enc(robot) # robot at time t
        a = self.action_enc(action) # action at time t
        h_pred = self.frame_predictor(cat([a, r, h], 1)) # pred. image at time t + 1
        x_pred = self.decoder([h_pred, skip])
        return x_pred, skip


class SVGModel(DeterministicModel):
    def __init__(self, config):
        self._config = config
        cf = config
        self.all_models = []
        input_dim = cf.action_enc_dim + cf.robot_enc_dim + cf.g_dim
        input_dim += cf.z_dim
        super().__init__(config, input_dim)
        self.posterior = post = GaussianLSTM(
            cf.robot_enc_dim + cf.g_dim,
            cf.z_dim,
            cf.rnn_size,
            cf.posterior_rnn_layers,
            cf.batch_size,
        ).to(self._device)

        self.prior = prior = GaussianLSTM(
            cf.action_enc_dim + cf.robot_enc_dim + cf.g_dim,
            cf.z_dim,
            cf.rnn_size,
            cf.prior_rnn_layers,
            cf.batch_size,
        ).to(self._device)
        self.all_models.extend([post, prior])

        # initialize weights and count parameters
        # def count_parameters(model):
        # return sum(p.numel() for p in model.parameters() if p.requires_grad)
        # num_p = 0
        for model in self.all_models:
            model.apply(init_weights)

    def init_hidden(self, batch_size=None):
        super().init_hidden(batch_size)
        self.posterior.hidden = self.posterior.init_hidden(batch_size)
        self.prior.hidden = self.prior.init_hidden(batch_size)


    def forward(self, image, mask, robot, action, next_image, next_mask, next_robot, skip=None):
        """Predict the next state using the learned prior or posterior
        If next_image, next_mask, next_robot are None, learned prior is used
        If skip is None, use image as skip

        Args:
            image ([Tensor]): batch of images
            mask ([Tensor]): batch of robot masks
            robot ([Tensor]): batch of robot states
            action ([Tensor]): batch of robot actions
            skip ([Tensor]): batch of skip connections
            use_posterior:

        Returns:
            [Tuple]: Next image, next latent, skip connection
        """
        h, curr_skip = self.encoder(cat([image, mask], dim=1))
        if skip is None:
            skip = curr_skip

        r = self.robot_enc(robot) # robot at time t
        a = self.action_enc(action) # action at time t

        # whether to use posterior or learned prior
        z_t, mu, logvar, mu_p, logvar_p = None, None, None, None, None
        z_p, mu_p, logvar_p = self.prior(cat([a, r, h], 1))
        z = z_p
        if next_image is not None:
            h_target = self.encoder(cat([next_image, next_mask], dim=1))[0]
            r_target = self.robot_enc(next_robot)
            z_t, mu, logvar = self.posterior(cat([r_target, h_target], 1))
            z = z_t
        h_pred = self.frame_predictor(cat([a, r, h, z], 1))
        x_pred = self.decoder([h_pred, skip])
        return x_pred, skip, mu, logvar, mu_p, logvar_p
