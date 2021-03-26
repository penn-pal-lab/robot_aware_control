from src.prediction.models.vgg_64 import CDNADecoder
import torch
import torch.nn as nn
from src.prediction.models.base import MLPEncoder, init_weights
from src.prediction.models.lstm import GaussianConvLSTM, RobonetConvLSTM, ConvLSTM, LSTM, GaussianLSTM
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


class DeterministicModel(nn.Module):
    def __init__(self, config, input_dim=None):
        super().__init__()
        self._config = cf = config
        self._device = config.device
        if input_dim is None:
            input_dim = cf.action_enc_dim + cf.g_dim
            if cf.model_use_robot_state:
                input_dim += cf.robot_enc_dim
        self._init_models(input_dim)

    def _init_models(self, input_dim):
        cf = self._config
        self.frame_predictor = frame_pred = LSTM(
            input_dim,
            cf.g_dim,
            cf.rnn_size,
            cf.predictor_rnn_layers,
            cf.batch_size,
        )

        if cf.image_width == 64:
            from src.prediction.models.vgg_64 import Decoder, Encoder
        elif cf.image_width == 128:
            from src.prediction.models.vgg import Decoder, Encoder

        channels = cf.channels
        if cf.model_use_mask:
            # RGB + mask channel
            channels += 1
            if cf.model_use_future_mask:
                channels += 1
        self.encoder = enc = Encoder(cf.g_dim, channels, cf.multiview, cf.dropout)
        self.decoder = dec = Decoder(cf.g_dim, cf.channels, cf.multiview)
        self.action_enc = ac = MLPEncoder(cf.action_dim, cf.action_enc_dim, 32)
        self.all_models = [frame_pred, enc, dec, ac]
        if cf.model_use_robot_state:
            self.robot_enc = MLPEncoder(cf.robot_dim, cf.robot_enc_dim, 32)
            self.all_models.append(self.robot_enc)

        self.to(self._device)
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
        cf = self._config
        if cf.model_use_mask:
            h, curr_skip = self.encoder(cat([image, mask], dim=1))
        else:
            h, curr_skip = self.encoder(image)

        if skip is None:
            skip = curr_skip

        a = self.action_enc(action)  # action at time t
        if cf.model_use_robot_state:
            r = self.robot_enc(robot)  # robot at time t
            h_pred = self.frame_predictor(
                cat([a, r, h], 1)
            )  # pred. image at time t + 1
        else:
            h_pred = self.frame_predictor(cat([a, h], 1))
        x_pred = self.decoder([h_pred, skip])
        return x_pred, skip


class SVGModel(DeterministicModel):
    def __init__(self, config):
        self._config = config
        cf = config
        self.all_models = []
        input_dim = cf.action_enc_dim + cf.g_dim + cf.z_dim
        post_dim = cf.g_dim
        prior_dim = cf.action_enc_dim + cf.g_dim
        if cf.model_use_robot_state:
            input_dim += cf.robot_enc_dim
            post_dim += cf.robot_enc_dim
            prior_dim += cf.robot_enc_dim
        super().__init__(config, input_dim)

        self.posterior = post = GaussianLSTM(
            post_dim,
            cf.z_dim,
            cf.rnn_size,
            cf.posterior_rnn_layers,
            cf.batch_size,
        )

        self.prior = prior = GaussianLSTM(
            prior_dim,
            cf.z_dim,
            cf.rnn_size,
            cf.prior_rnn_layers,
            cf.batch_size,
        )
        self.all_models.extend([post, prior])
        self.to(self._device)
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

    def forward(
        self, image, mask, robot, action, next_image, next_mask, next_robot, skip=None, force_use_prior=False
    ):
        """Predict the next state using the learned prior or posterior
        If next_image, next_mask, next_robot are None, learned prior is used
        If skip is None, use current image as skip

        Args:
            image ([Tensor]): batch of images
            mask ([Tensor]): batch of robot masks
            robot ([Tensor]): batch of robot states
            action ([Tensor]): batch of robot actions
            skip ([Tensor]): batch of skip connections
            force_use_prior: use prior z even if posterior z is computed. Use this when we want to use the prior, but need to compute statistics about the posterior z

        Returns:
            [Tuple]: Next image, next latent, skip connection
        """
        cf = self._config
        if cf.model_use_mask:
            h, curr_skip = self.encoder(cat([image, mask], dim=1))
        else:
            h, curr_skip = self.encoder(image)
        if skip is None:
            skip = curr_skip

        a = self.action_enc(action)  # action at time t
        z_t, mu, logvar, mu_p, logvar_p = None, None, None, None, None
        if cf.model_use_robot_state:
            r = self.robot_enc(robot)  # robot at time t
            # whether to use posterior or learned prior
            z_p, mu_p, logvar_p = self.prior(cat([a, r, h], 1))
        else:
            z_p, mu_p, logvar_p = self.prior(cat([a, h], 1))

        z = z_p
        if next_image is not None:
            if cf.model_use_mask:
                h_target = self.encoder(cat([next_image, next_mask], dim=1))[0]
            else:
                h_target = self.encoder(next_image)[0]
            if cf.model_use_robot_state:
                r_target = self.robot_enc(next_robot)
                z_t, mu, logvar = self.posterior(cat([r_target, h_target], 1))
            else:
                z_t, mu, logvar = self.posterior(h_target)
            if not force_use_prior:
                z = z_t

        if cf.model_use_robot_state:
            h_pred = self.frame_predictor(cat([a, r, h, z], 1))
        else:
            h_pred = self.frame_predictor(cat([a, h, z], 1))

        x_pred = self.decoder([h_pred, skip])
        return x_pred, skip, mu, logvar, mu_p, logvar_p


class JointPosPredictor(nn.Module):
    """
    Predicts the next joint pos of the robot
    """
    def __init__(self, config):
        super().__init__()

        input_dim = config.robot_joint_dim + config.action_dim
        output_dim = config.robot_joint_dim
        hidden_size = 512
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, output_dim),
        )

    def forward(self, joints, action):
        """Predict the next joint position

        Args:
            joints (Tensor): Tensor of joint positions
            action (Tensor): Tensor of end effector displacements

        Returns:
            Tensor: difference in joint positions between future and current joint position after applying action
        """
        input = cat([joints, action], 1)
        out = self.predictor(input)
        return out

class GripperStatePredictor(nn.Module):
    """
    Predicts the next eef pos, vel, grip of the robot
    """
    def __init__(self, config):
        super().__init__()

        input_dim = config.robot_dim + config.action_dim
        output_dim = config.robot_dim
        hidden_size = 512
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, output_dim),
        )

    def forward(self, eef_pose, action):
        """Predict the next eef pose position

        Args:
            eef_pose (Tensor): Tensor of end effector poses
            action (Tensor): Tensor of end effector displacements

        Returns:
            Tensor: difference in eef pose between future and current eef pose after applying action
        """
        input = cat([eef_pose, action], 1)
        out = self.predictor(input)
        return out


class CopyModel(nn.Module):
    """
    Baseline that copies the previous image  to predict the next image
    """

    @torch.no_grad()
    def forward(self, image, mask, next_image, next_mask):
        image = image.detach().clone()
        next_image = next_image.detach().clone()
        next_mask = next_mask.type(torch.bool).repeat(1, 3, 1, 1)

        # set world pixels of next image to the world pixels of the previous image
        next_image[~next_mask] = image[~next_mask]
        # world pixels of next image that are occupied by robot in previous image just get set to next image world pixels
        # next_image[~next_mask & mask] = image[~next_mask & mask]

        return next_image

    def init_hidden(self, batch_size=None):
        pass



class DeterministicConvModel(nn.Module):
    """Deterministic Conv LSTM predictor
    No longer need robot encoder, action encoder, just tile directly.
    """
    def __init__(self, config):
        super().__init__()
        self._config = cf = config
        self._device = config.device
        self._image_width = cf.image_width
        self._image_height = cf.image_height
        self._init_models()

    def _init_models(self):
        cf = self._config
        if cf.image_width == 64:
            from src.prediction.models.vgg_64 import ConvDecoder, ConvEncoder
        else:
            raise ValueError

        channels = cf.channels
        if cf.model_use_mask:
            # RGB + mask channel
            channels += 1
            if cf.model_use_future_mask:
                channels += 1
        self.encoder = enc = ConvEncoder(cf.g_dim, channels)

        # 2 channel spatial map for actions
        height, width = self._image_height // 8, self._image_width // 8
        self.action_encoder = ac_enc =  nn.Sequential(
            nn.Linear(cf.action_dim, height * width * 2)
        )
        # 2 channel spatial map for state
        if cf.model_use_robot_state:
            self.state_encoder = state_enc = nn.Sequential(
                nn.Linear(cf.robot_dim, height * width * 2)
            )

        # tile robot state, action convolution
        in_channels = cf.g_dim + 2 + (2 * cf.model_use_robot_state)
        self.frame_predictor = frame_pred = ConvLSTM(cf, in_channels)
        self.decoder = dec = ConvDecoder(in_channels, cf.channels + 1) # extra channel for attention
        self.all_models = [frame_pred, enc, dec, ac_enc]
        if cf.model_use_robot_state:
            self.all_models.append(state_enc)

        self.to(self._device)
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
        cf = self._config
        if cf.model_use_mask:
            h, curr_skip = self.encoder(cat([image, mask], dim=1))
        else:
            h, curr_skip = self.encoder(image)

        if skip is None:
            skip = curr_skip
        # TODO: pass through MLP and then reshape to 2d map
        # tile the action and states
        height, width = self._image_height // 8, self._image_width // 8
        ac_enc = self.action_encoder(action).view(action.shape[0], 2, height, width)
        if cf.model_use_robot_state:
            r = self.state_encoder(robot).view(robot.shape[0], 2, height, width)
            state = torch.cat([h, ac_enc, r], 1)
        else:
            state = torch.cat([h, ac_enc], 1)
        h_pred = self.frame_predictor(state)
        x_pred = self.decoder([h_pred, skip])
        return x_pred, skip


class SVGConvModel(nn.Module):
    """Conv SVG LSTM predictor
    """
    def __init__(self, config):
        super().__init__()
        self._config = cf = config
        self._device = config.device
        self._image_width = cf.image_width
        self._image_height = cf.image_height
        self._init_models()

    def _init_models(self):
        cf = self._config
        if cf.image_width == 64:
            from src.prediction.models.vgg_64 import ConvDecoder, ConvEncoder
        else:
            raise ValueError

        channels = cf.channels
        if cf.model_use_mask:
            # RGB + mask channel
            channels += 1
            if cf.model_use_future_mask:
                channels += 1

        if cf.model_use_heatmap:
            channels += 1
            if cf.model_use_future_heatmap:
                channels += 1

        self.encoder = enc = ConvEncoder(cf.g_dim, channels)

        # 2 channel spatial map for actions
        height, width = self._image_height // 8, self._image_width // 8
        self.action_encoder = ac_enc =  nn.Sequential(
            nn.Linear(cf.action_dim, height * width * 2)
        )
        # 2 channel spatial map for state
        if cf.model_use_robot_state:
            self.state_encoder = state_enc = nn.Sequential(
                nn.Linear(cf.robot_dim, height * width * 2)
            )
        if cf.model_use_future_robot_state:
            self.future_state_encoder = fut_state_enc = nn.Sequential(
                nn.Linear(cf.robot_dim, height * width * 2)
            )
        post_dim = cf.g_dim
        prior_dim = cf.g_dim + 2 # ac channels
        if cf.model_use_robot_state:
            post_dim += 2
            prior_dim += 2
        if cf.model_use_future_robot_state:
            # posterior doesn't need 2 extra channels for future robot state,
            # since it only needs the r_i to produce z.
            prior_dim += 2

        self.posterior = post = GaussianConvLSTM(cf, post_dim, cf.z_dim)
        self.prior = prior = GaussianConvLSTM(cf, prior_dim, cf.z_dim)

        # encoder channels, noise channels, ac channels, state channels
        in_channels = cf.g_dim + cf.z_dim + 2 + (2 * cf.model_use_robot_state) + (2 * cf.model_use_future_robot_state)
        self.frame_predictor = frame_pred = ConvLSTM(cf, in_channels)
        self.decoder = dec = ConvDecoder(in_channels, cf.channels + 1) # extra channel for attention

        self.all_models = [frame_pred, enc, dec, ac_enc, post, prior]
        if cf.model_use_robot_state:
            self.all_models.append(state_enc)
        if cf.model_use_future_robot_state:
            self.all_models.append(fut_state_enc)

        self.to(self._device)
        for model in self.all_models:
            model.apply(init_weights)

    def init_hidden(self, batch_size=None):
        """
        Initialize the recurrent states by batch size
        """
        self.frame_predictor.hidden = self.frame_predictor.init_hidden(batch_size)
        self.posterior.hidden = self.posterior.init_hidden(batch_size)
        self.prior.hidden = self.prior.init_hidden(batch_size)

    def forward(
        self, image, mask, robot, heatmap, action, next_image, next_mask, next_robot, next_heatmap, skip=None, force_use_prior=False
    ):
        """Predict the next state using the learned prior or posterior
        If next_image, next_mask, next_robot are None, learned prior is used
        If skip is None, use current image as skip

        Args:
            image ([Tensor]): batch of images
            mask ([Tensor]): batch of robot masks
            robot ([Tensor]): batch of robot states
            action ([Tensor]): batch of robot actions
            skip ([Tensor]): batch of skip connections
            force_use_prior: use prior z even if posterior z is computed. Use this when we want to use the prior, but need to compute metrics about the posterior z for training.

        Returns:
            [Tuple]: Next image, next latent, skip connection
        """
        cf = self._config

        # combine image, mask, heatmap, in channel dimension
        img = image
        if cf.model_use_heatmap:
            img = cat([img, heatmap], dim=1)
        if cf.model_use_mask:
            img = cat([img, mask], dim=1)

        h, curr_skip = self.encoder(img)

        if cf.last_frame_skip or skip is None:
            # use the current image's skip to decoder
            skip = curr_skip

        # tile the action and states
        height, width = self._image_height // 8, self._image_width // 8
        a = self.action_encoder(action).view(action.shape[0], 2, height, width)
        z_t, mu, logvar, mu_p, logvar_p = None, None, None, None, None

        if cf.model_use_robot_state:
            if cf.model_use_future_robot_state:
                r, r_next = robot
                r = self.state_encoder(r).view(r.shape[0], 2, height, width)
                r_next = self.future_state_encoder(r_next).view(r_next.shape[0], 2, height, width)
                z_p, mu_p, logvar_p = self.prior(cat([a, r, r_next, h], 1))
            else:
                r = self.state_encoder(robot).view(robot.shape[0], 2, height, width)
                # whether to use posterior or learned prior
                z_p, mu_p, logvar_p = self.prior(cat([a, r, h], 1))
        else:
            z_p, mu_p, logvar_p = self.prior(cat([a, h], 1))
        # use prior's z by default
        z = z_p

        # if future image is supplied, calculate the posterior
        if next_image is not None:
            next_img = next_image
            if cf.model_use_heatmap:
                next_img = cat([next_img, next_heatmap], dim=1)
            if cf.model_use_mask:
                next_img = cat([next_img, next_mask], dim=1)
            h_target = self.encoder(img)[0]

            if cf.model_use_robot_state:
                r_target = self.state_encoder(next_robot).view(next_robot.shape[0], 2, height, width)
                z_t, mu, logvar = self.posterior(cat([r_target, h_target], 1))
            else:
                z_t, mu, logvar = self.posterior(h_target)
            if not force_use_prior:
                z = z_t

        # add z to the latent before prediction
        if cf.model_use_robot_state:
            if cf.model_use_future_robot_state:
                h_pred = self.frame_predictor(cat([a, r, r_next, h, z], 1))
            else:
                h_pred = self.frame_predictor(cat([a, r, h, z], 1))
        else:
            h_pred = self.frame_predictor(cat([a, h, z], 1))

        x_pred = self.decoder([h_pred, skip])
        return x_pred, skip, mu, logvar, mu_p, logvar_p

class DeterministicCDNAModel(nn.Module):
    """Deterministic CDNA LSTM predictor
    """
    def __init__(self, config):
        super().__init__()
        self._config = cf = config
        self._device = config.device
        self._init_models()

    def _init_models(self):
        cf = self._config

        hid_channels = cf.g_dim
        self.frame_predictor = frame_pred = ConvLSTM(
            cf.batch_size,
            hid_channels
        )

        if cf.image_width == 64:
            from src.prediction.models.vgg_64 import ConvEncoder
        else:
            raise ValueError

        channels = cf.channels
        if cf.model_use_mask:
            # RGB + mask channel
            channels += 1
            if cf.model_use_future_mask:
                channels += 1
        self.encoder = enc = ConvEncoder(cf.g_dim, channels)
        # tile robot state, action convolution
        in_channels = cf.g_dim + cf.action_dim
        if cf.model_use_robot_state:
            in_channels += cf.robot_dim

        self.tiled_state_conv = tsv = nn.Conv2d(in_channels, cf.g_dim, 3,1,1)
        self.decoder = dec = CDNADecoder(cf.g_dim, cf.cdna_kernel_size)
        self.all_models = [frame_pred, enc, dec, tsv]

        self.to(self._device)
        for model in self.all_models:
            model.apply(init_weights)

    def init_hidden(self, batch_size=None):
        """
        Initialize the recurrent states by batch size
        """
        self.frame_predictor.hidden = self.frame_predictor.init_hidden(batch_size)

    def forward(self, image, mask, robot, action, context_image, skip=None):
        """Given current state, predict the next state

        Args:
            image ([Tensor]): batch of images
            mask ([Tensor]): batch of robot masks
            robot ([Tensor]): batch of robot states
            action ([Tensor]): batch of robot actions
            context_image ([Tensor]): batch of ground truth image
            skip ([Tensor]): batch of skip connections

        Returns:
            [Tuple]: Next image, next latent, skip connection
        """
        cf = self._config
        if cf.model_use_mask:
            h, curr_skip = self.encoder(cat([image, mask], dim=1))
        else:
            h, curr_skip = self.encoder(image)

        if skip is None:
            skip = curr_skip
        # tile the action and states
        a = action.repeat(8, 8, 1, 1).permute(2,3,0,1)
        if cf.model_use_robot_state:
            r = robot.repeat(8,8,1,1).permute(2,3,0,1)
            tiled_state = cat([a, r, h], 1)
        else:
            tiled_state = cat([a, h], 1)
        state = self.tiled_state_conv(tiled_state)
        h_pred = self.frame_predictor(state)
        x_pred = self.decoder(image, h_pred, skip, context_image)
        return x_pred, skip



class RobonetCDNAModel(nn.Module):
    """Enc -> Enc LSTM -> attention -> Dec LSTM ->  CDNA Decoder
    """
    def __init__(self, config):
        super().__init__()
        self._config = cf = config
        self._device = config.device
        self._init_models()

    def _init_models(self):
        cf = self._config
        if cf.image_width == 64:
            from src.prediction.models.vgg_64 import ConvEncoder
        else:
            raise ValueError

        channels = cf.channels
        if cf.model_use_mask:
            # RGB + mask channel
            channels += 1
            if cf.model_use_future_mask:
                channels += 1
        self.encoder = enc = ConvEncoder(cf.g_dim, channels)

        # 2 channel spatial map for actions
        self.action_encoder = ac_enc =  nn.Sequential(
            nn.Linear(cf.action_dim, 8 * 8 * 2)
        )
        # 2 channel spatial map for state
        if cf.model_use_robot_state:
            self.state_encoder = state_enc = nn.Sequential(
                nn.Linear(cf.robot_dim, 8 * 8 * 2)
            )

        # tile robot state, action convolution
        in_channels = cf.g_dim + 2 + (2 * cf.model_use_robot_state)
        self.inst_norm = nn.InstanceNorm2d(in_channels)

        self.frame_predictor = frame_pred = RobonetConvLSTM(
            cf.batch_size,
            in_channels
        )

        self.decoder = dec = CDNADecoder(in_channels, cf.cdna_kernel_size)

        self.all_models = [frame_pred, enc, dec, ac_enc, self.inst_norm]
        if cf.model_use_robot_state:
            self.all_models.append(state_enc)

        self.to(self._device)
        for model in self.all_models:
            model.apply(init_weights)

    def init_hidden(self, batch_size=None):
        """
        Initialize the recurrent states by batch size
        """
        self.frame_predictor.hidden = self.frame_predictor.init_hidden(batch_size)

    def forward(self, image, mask, robot, action, context_image, skip=None):
        """Given current state, predict the next state

        Args:
            image ([Tensor]): batch of images
            mask ([Tensor]): batch of robot masks
            robot ([Tensor]): batch of robot states
            action ([Tensor]): batch of robot actions
            context_image ([Tensor]): batch of ground truth image
            skip ([Tensor]): batch of skip connections

        Returns:
            [Tuple]: Next image, next latent, skip connection
        """
        cf = self._config
        if cf.model_use_mask:
            h, curr_skip = self.encoder(cat([image, mask], dim=1))
        else:
            h, curr_skip = self.encoder(image)

        if skip is None:
            skip = curr_skip
        ac_enc = self.action_encoder(action).view(action.shape[0], 2, 8, 8)
        if cf.model_use_robot_state:
            r = self.state_encoder(robot).view(robot.shape[0], 2, 8, 8)
            state = torch.cat([h, ac_enc, r], 1)
        else:
            state = torch.cat([h, ac_enc], 1)
        state = self.inst_norm(state)
        h_pred = self.frame_predictor(state)
        x_pred = self.decoder(image, h_pred, context_image)
        return x_pred, skip