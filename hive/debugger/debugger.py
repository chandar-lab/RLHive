from pathlib import Path
from hive.utils.registry import Registrable
from hive.debugger.utils import settings


# TODO: Debugger class needs to be divided in multiple classes :
#       1- Debugger class abstract class with the abstract methods
#       2- NullDebugger class (following the logger in RLHive logic) when the user doesn't want to use the Debugger
#       3- PreCheckDebugger class to run the pre-check properties
#       4- PostCheckDebugger class to run the post-check properties
#       5- OnTrainingCheckDebugger class to run the on-training-check properties
#       6- CompositeDebugger class to run the all-check properties
class Debugger(Registrable):
    def __init__(self, check_type, app_path=None):
        # TODO: merge this logger with RLHive's logger
        app_path = Path.cwd() if app_path == None else app_path
        log_fpath = settings.build_log_file_path(app_path, check_type)
        self.logger = settings.file_logger(log_fpath, check_type)
        # TODO: we need to follow the same config logic as RLHive !!!!
        config_fpath = settings.load_user_config_if_exists(app_path)
        self.config = settings.Config(config_fpath)
        self.pre_check = PreCheck(main_logger=self.logger, config=self.config.pre_check)

    @classmethod
    def type_name(cls):
        return "debugger"

    def set_parameters(self, observations, model, labels, predictions, loss, opt, actions):
        self.observations = observations
        self.model = model
        self.labels = labels
        self.predictions = predictions
        self.loss = loss
        self.opt = opt
        self.actions = actions
        self.pre_check.set_parameters(observations, model, labels, predictions, loss, opt, actions)

    def run_pre_checks(self, batch_size, implemented_ops):
        self.pre_check.run(batch_size, implemented_ops)


# TODO: This class should be named "PreCheckDebugger"
class PreCheck:

    def __init__(self, main_logger, config):
        # this parameters prevents running the pre_check pultiple times
        self.pre_check_done = False
        self.main_logger = main_logger
        self.config = config
        self.main_msgs = settings.load_messages()

    def set_parameters(self, observations, model, labels, predictions, loss, opt, actions):
        self.observations = observations
        self.model = model
        self.labels = labels
        self.predictions = predictions
        self.loss = loss
        self.opt = opt
        self.actions = actions

    def react(self, message):
        if self.config.fail_on:
            self.main_logger.error(message)
            raise Exception(message)
        else:
            self.main_logger.warning(message)

    def _pre_check_observations(self):
        mas = np.max(self.observations)
        mis = np.min(self.observations)
        avgs = np.mean(self.observations)
        stds = np.std(self.observations)

        # for idx in range(len(mas)):
        if stds == 0.0:
            msg = self.main_msgs['features_constant']
            self.react(msg)
        elif any([almost_equal(mas, data_max) for data_max in self.config.data.normalized_data_maxs]) and \
                any([almost_equal(mis, data_min) for data_min in self.config.data.normalized_data_mins]):
            return
        elif not (almost_equal(stds, 1.0) and almost_equal(avgs, 0.0)):
            msg = self.main_msgs['features_unnormalized']
            self.react(msg)

    def _pre_check_weights(self):
        # a = self.model.base_network._modules
        initial_weights, _ = self.get_model_weights_and_biases()
        layer_names = self.get_model_layer_names()
        for layer_name, weight_array in initial_weights.items():
            shape = weight_array.shape
            if len(shape) == 1 and shape[0] == 1:
                continue
            if almost_equal(np.var(weight_array), 0.0, rtol=1e-8):
                self.react(self.main_msgs['poor_init'].format(layer_name))
            else:
                if len(shape) == 2:
                    fan_in = shape[0]
                    fan_out = shape[1]
                else:
                    receptive_field_size = np.prod(shape[:-2])
                    fan_in = shape[-2] * receptive_field_size
                    fan_out = shape[-1] * receptive_field_size
                lecun_F, lecun_test = metrics.pure_f_test(weight_array, np.sqrt(1.0 / fan_in),
                                                          self.config.init_w.f_test_alpha)
                he_F, he_test = metrics.pure_f_test(weight_array, np.sqrt(2.0 / fan_in),
                                                    self.config.init_w.f_test_alpha)
                glorot_F, glorot_test = metrics.pure_f_test(weight_array, np.sqrt(2.0 / (fan_in + fan_out)),
                                                            self.config.init_w.f_test_alpha)
                try:
                    activation_layer = list(layer_names)[list(layer_names.keys()).index(layer_name) + 1]
                except Exception:
                    continue
                if isinstance(layer_names[activation_layer], torch.nn.ReLU) and not he_test:
                    abs_std_err = np.abs(np.std(weight_array) - np.sqrt(1.0 / fan_in))
                    self.react(self.main_msgs['need_he'].format(layer_name, abs_std_err))
                elif isinstance(layer_names[activation_layer], torch.nn.Tanh) and not glorot_test:
                    abs_std_err = np.abs(np.std(weight_array) - np.sqrt(2.0 / fan_in))
                    self.react(self.main_msgs['need_glorot'].format(layer_name, abs_std_err))
                elif isinstance(layer_names[activation_layer], torch.nn.Sigmoid) and not lecun_test:
                    abs_std_err = np.abs(np.std(weight_array) - np.sqrt(2.0 / (fan_in + fan_out)))
                    self.react(self.main_msgs['need_lecun'].format(layer_name, abs_std_err))
                elif not (lecun_test or he_test or glorot_test):
                    self.react(self.main_msgs['need_init_well'].format(layer_name))

    def _pre_check_biases(self):
        _, initial_biases = self.get_model_weights_and_biases()
        if not (initial_biases):
            self.react(self.main_msgs['need_bias'])
        else:
            checks = []
            for b_name, b_array in initial_biases.items():
                checks.append(np.sum(b_array) == 0.0)
            if not np.all(checks):
                self.react(self.main_msgs['zero_bias'])

    def _pre_check_loss(self):
        losses = []
        n = self.config.init_loss.size_growth_rate
        while n <= (self.config.init_loss.size_growth_rate * self.config.init_loss.size_growth_iters):
            derived_batch_y = np.concatenate(n * [self.labels], axis=0)
            derived_predictions = np.concatenate(n * [numpify(self.predictions)], axis=0)
            loss = float(self.get_loss(derived_predictions, derived_batch_y))
            losses.append(loss)
            n *= self.config.init_loss.size_growth_rate
        rounded_loss_rates = [round(losses[i + 1] / losses[i]) for i in range(len(losses) - 1)]
        # rounded_loss_rates = [losses[i + 1] / losses[i] for i in range(len(losses) - 1)]
        equality_checks = sum(
            [(loss_rate == self.config.init_loss.size_growth_rate) for loss_rate in rounded_loss_rates])
        if equality_checks == len(rounded_loss_rates):
            self.react(self.main_msgs['poor_reduction_loss'])
        initial_loss = float(self.get_loss(self.predictions, self.labels))
        # specify here the number of actions
        number_of_actions = 100000
        expected_loss = -np.log(1 / number_of_actions)
        err = np.abs(initial_loss - expected_loss)
        if err >= self.config.init_loss.dev_ratio * expected_loss:
            self.react(self.main_msgs['poor_init_loss'].format(round((err / expected_loss), 3)))

    def _pre_check_gradients(self):
        # Train the model on the data we have
        all_weights, _ = self.get_model_weights_and_biases()
        weights_shapes = [[int(s) for s in list(weight.shape)] for weight in all_weights]
        few_x, few_y = self.get_sample(self.observations, self.labels, self.config.grad.sample_size)
        for i in range(len(all_weights)):
            theoretical, numerical = tf.test.compute_gradient(
                all_weights[i],
                weights_shapes[i],
                self.nn_data.model.loss,
                [1],
                delta=self.config.grad.delta,
                x_init_value=init_weights[i],
                extra_feed_dict=feed_dict
            )
            theoretical, numerical = theoretical.flatten(), numerical.flatten()
            total_dims, sample_dims = len(theoretical), int(self.config.grad.ratio_of_dimensions * len(theoretical))
            indices = np.random.choice(np.arange(total_dims), sample_dims, replace=False)
            theoretical_sample = theoretical[indices]
            numerical_sample = numerical[indices]
            numerator = np.linalg.norm(theoretical_sample - numerical_sample)
            denominator = np.linalg.norm(theoretical_sample) + np.linalg.norm(numerical_sample)
            relerr = numerator / denominator
            if relerr > self.config.grad.relative_err_max_thresh:
                self.react(self.main_msgs['grad_err'].format(all_weights[i], readable(relerr),
                                                             self.config.grad.relative_err_max_thresh))

    def _pre_check_fitting_data_capability(self):

        def _loss_is_stable(loss_value):
            if np.isnan(loss_value):
                self.react(self.main_msgs['nan_loss'])
                return False
            if np.isinf(loss_value):
                self.react(self.main_msgs['inf_loss'])
                return False
            return True

        derived_batch_x = self.observations
        derived_batch_y = self.labels

        real_losses = []
        for _ in range(self.config.prop_fit.total_iters):

            self.opt.zero_grad()
            outputs = self.model(torch.tensor(derived_batch_x))
            outputs = outputs[torch.arange(outputs.size(0)), self.actions]
            loss = self.get_loss(outputs, derived_batch_y)
            loss.backward()
            self.opt.step()
            real_losses.append(loss.item())

            if not (_loss_is_stable(loss.item())):
                self.react(self.main_msgs['underfitting_single_batch'])
                return

        loss_smoothness = metrics.smoothness(np.array(real_losses))
        min_loss = np.min(np.array(real_losses))
        if min_loss <= self.config.prop_fit.abs_loss_min_thresh or (
                min_loss <= self.config.prop_fit.loss_min_thresh and
                loss_smoothness > self.config.prop_fit.smoothness_max_thresh):
            self.react(self.main_msgs['zero_loss'])
            return
        # if not (underfitting_prob): return
        zeroed_batch_x = np.zeros_like(derived_batch_x)
        fake_losses = []
        for _ in range(self.config.prop_fit.total_iters):
            outputs = self.model(torch.tensor(zeroed_batch_x))
            outputs = outputs[torch.arange(outputs.size(0)), self.actions]
            fake_loss = float(self.get_loss(outputs, derived_batch_y))
            fake_losses.append(fake_loss)
            if not (_loss_is_stable(fake_loss)): return
        stability_test = np.array([_loss_is_stable(loss_value) for loss_value in (real_losses + fake_losses)])
        if (stability_test == False).any():
            last_real_losses = real_losses[-self.config.prop_fit.sample_size_of_losses:]
            last_fake_losses = fake_losses[-self.config.prop_fit.sample_size_of_losses:]
            if not (metrics.are_significantly_different(last_real_losses, last_fake_losses)):
                self.react(self.main_msgs['data_dep'])

    def run(self, batch_size=None, implemented_ops=None):
        self.pre_check_done = True
        if self.config.disabled:
            return
        self._pre_check_observations()
        self._pre_check_weights()
        self._pre_check_biases()
        self._pre_check_loss()
        self._pre_check_fitting_data_capability()
        # self._pre_check_gradients()

    def get_model_weights_and_biases(self):
        weights = {}
        biases = {}
        i = 0
        for key, value in self.model.state_dict().items():
            if i % 2 == 0:
                weights[str(key.split('.weight')[0])] = value.numpy()
            else:
                biases[str(key.split('.bias')[0])] = value.numpy()
            i += 1

        return weights, biases

    def get_sample(self, size):
        pass

    def get_loss(self, original_predictions, model_predictions):
        loss = self.loss(torch.Tensor(original_predictions), torch.Tensor(model_predictions)).mean()
        return loss

    def get_accuracy(self, batch_x, batch_y):

        pass

    def get_model_layer_names(self):
        layer_names = OrderedDict()
        for name, layer in self.model.named_modules():
            layer_names[name] = layer
        return layer_names

    def get_sample(self, observations, labels, sample_size):
        return observations[:sample_size], labels[:sample_size]


def almost_equal(value1, value2, rtol=1e-2):
    rerr = np.abs(value1 - value2)
    if isinstance(value1, np.ndarray):
        return (rerr <= rtol).all()
    else:
        return rerr <= rtol



