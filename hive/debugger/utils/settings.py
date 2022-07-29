import os
import logging
import yaml
from fastnumbers import fast_real
from configparser import ConfigParser
from pathlib import Path 

#Conventional constants
LOG_DIR = 'logs'
CONFIG_DIR = 'config'
CLASSIFICATION_KEY = 'classification'
REGRESSION_KEY = 'regression'
EPSILON = 1e-16

#Standard messages
def load_messages(messages_section='Messages'):
    messages_fpath = os.path.join(os.path.join(Path(__file__).parent,'config'), 'messages.properties')
    config = ConfigParser()
    config.read(messages_fpath)
    return dict(config[messages_section])

#Configuration 
def load_user_config_if_exists(app_path):
    user_config_filepath = None
    config_dir_path = os.path.join(app_path, CONFIG_DIR)
    if os.path.exists(config_dir_path):
        for filename in os.listdir(config_dir_path):
            if filename.endswith(".yaml") and ('settings' in filename or 'config' in filename):
                user_config_filepath = os.path.join(config_dir_path, filename)
                break
    return user_config_filepath

class Config:

    def __init__(self, user_config_fpath=None, standard_config_fpath=None):
        if standard_config_fpath == None:
            standard_config_fpath = os.path.join(os.path.join(Path(__file__).parent,'config'), 'settings.yaml')
        standard_config_dict = yaml.load(open(standard_config_fpath, 'r'), Loader=yaml.SafeLoader)
        if user_config_fpath != None:
            user_config_dict = yaml.load(open(user_config_fpath, 'r'), Loader=yaml.SafeLoader)
            Config.override_dict(standard_config_dict, user_config_dict)
        else:
            Config.override_dict(standard_config_dict)
        self.pre_check = Config.PreCheck.build(standard_config_dict['PreCheck'])
        self.overfit_check = Config.OverfitCheck.build(standard_config_dict['OverfitCheck'])
        self.post_check = Config.PostCheck.build(standard_config_dict['PostCheck'])

    @staticmethod
    def override_dict(standard_dict, user_dict=None):
        for key, value in standard_dict.items():
            if isinstance(value, dict):
                if user_dict is not None and key in user_dict:
                    Config.override_dict(value, user_dict[key])
                else:
                    Config.override_dict(value)
            else:
                if user_dict is not None and key in user_dict:
                    standard_dict[key] = user_dict[key]
    
    class PreCheck:

        def __init__(self, fail_on, disabled, data, init_w, init_b, init_loss, grad, prop_fit, ins_wise_op):
            self.fail_on = fail_on
            self.disabled = disabled
            self.data = data
            self.init_loss = init_loss
            self.init_w = init_w
            self.init_b = init_b
            self.grad = grad
            self.prop_fit = prop_fit
            self.ins_wise_op = ins_wise_op
            
        @staticmethod
        def build(pre_check_config):
            fail_on, disabled = pre_check_config['fail_on'], pre_check_config['disabled']
            data = Config.PreCheck.Data.build(disabled, pre_check_config['Data'])
            init_loss = Config.PreCheck.Initial_Loss.build(disabled, pre_check_config['Initial_Loss'])
            init_w = Config.PreCheck.Initial_Weight.build(disabled, pre_check_config['Initial_Weight'])
            init_b = Config.PreCheck.Initial_Bias.build(disabled, pre_check_config['Initial_Bias'])
            grad = Config.PreCheck.Gradient.build(disabled, pre_check_config['Gradient'])
            prop_fit = Config.PreCheck.Proper_Fitting.build(disabled, pre_check_config['Proper_Fitting'])
            ins_wise_op = Config.PreCheck.Instance_wise_Operation.build(disabled, pre_check_config['Instance_wise_Operation']) 
            return Config.PreCheck(fail_on, disabled, data, init_w, init_b, init_loss, grad, prop_fit, ins_wise_op)

        class Data: 
            def __init__(self, disabled, normalized_data_mins, normalized_data_maxs, outputs_var_coef_thresh, labels_perp_min_thresh):
                self.disabled = disabled
                self.normalized_data_mins = [fast_real(min_elt) for min_elt in normalized_data_mins]
                self.normalized_data_maxs = [fast_real(max_elt) for max_elt in normalized_data_maxs]
                self.outputs_var_coef_thresh = fast_real(outputs_var_coef_thresh)
                self.labels_perp_min_thresh = fast_real(labels_perp_min_thresh)
            @staticmethod
            def build(disabled, data_config):
                disabled = (disabled or data_config['disabled']) if 'disabled' in data_config else disabled
                return Config.PreCheck.Data(disabled, data_config['normalized_data_mins'], data_config['normalized_data_maxs'],
                                                      data_config['outputs_var_coef_thresh'], data_config['labels_perp_min_thresh'])

        class Initial_Weight:
            def __init__(self, disabled, f_test_alpha):
                self.disabled = disabled
                self.f_test_alpha = f_test_alpha

            @staticmethod
            def build(disabled, init_w_config):
                disabled = (disabled or init_w_config['disabled']) if 'disabled' in init_w_config else disabled
                f_test_alpha = init_w_config['f_test_alpha']
                return Config.PreCheck.Initial_Weight(disabled, f_test_alpha)
        
        class Initial_Bias:
            def __init__(self, disabled):
                self.disabled = disabled

            @staticmethod
            def build(disabled, init_b_config):
                disabled = (disabled or init_b_config['disabled']) if 'disabled' in init_b_config else disabled
                return Config.PreCheck.Initial_Bias(disabled)

        class Initial_Loss:
            def __init__(self, disabled, sample_size, size_growth_rate, size_growth_iters, dev_ratio):
                self.disabled = disabled
                self.sample_size = fast_real(sample_size)
                self.dev_ratio = dev_ratio
                self.size_growth_rate = size_growth_rate
                self.size_growth_iters = size_growth_iters
            @staticmethod
            def build(disabled, init_loss_config):
                disabled = (disabled or init_loss_config['disabled']) if 'disabled' in init_loss_config else disabled
                sample_size = init_loss_config['sample_size']
                size_growth_rate = init_loss_config['data_size_growth_rate']
                size_growth_iters = init_loss_config['data_size_growth_iters']
                dev_ratio = init_loss_config['dev_ratio']
                return Config.PreCheck.Initial_Loss(disabled, sample_size, size_growth_rate, size_growth_iters, dev_ratio)
        
        class Gradient:
            def __init__(self, disabled, warm_up_steps, warm_up_batch, sample_size, ratio_of_dimensions, delta, relative_err_max_thresh):
                self.disabled = disabled
                self.warm_up_steps = fast_real(warm_up_steps)
                self.warm_up_batch = fast_real(warm_up_batch)
                self.sample_size = fast_real(sample_size)
                self.ratio_of_dimensions = fast_real(ratio_of_dimensions)
                self.delta = fast_real(delta)
                self.relative_err_max_thresh = fast_real(relative_err_max_thresh)
            
            @staticmethod
            def build(disabled, grad_config):
                disabled = (disabled or grad_config['disabled']) if 'disabled' in grad_config else disabled
                warm_up_steps = grad_config['warm_up_steps']
                warm_up_batch = grad_config['warm_up_batch']
                sample_size = grad_config['sample_size']
                ratio_of_dimensions = grad_config['ratio_of_dimensions']
                delta = grad_config['delta']
                relative_err_max_thresh = grad_config['relative_err_max_thresh']
                return Config.PreCheck.Gradient(disabled, warm_up_steps, warm_up_batch, sample_size, ratio_of_dimensions, delta, relative_err_max_thresh)
        
        class Proper_Fitting:
            def __init__(self, disabled, single_batch_size, total_iters, abs_loss_min_thresh, loss_min_thresh, smoothness_max_thresh, mislabeled_rate_max_thresh, mean_error_max_thresh, sample_size_of_losses):
                self.disabled = disabled
                self.single_batch_size = fast_real(single_batch_size)
                self.total_iters = fast_real(total_iters)
                self.abs_loss_min_thresh = fast_real(abs_loss_min_thresh)
                self.loss_min_thresh = fast_real(loss_min_thresh)
                self.smoothness_max_thresh = fast_real(smoothness_max_thresh)
                self.mislabeled_rate_max_thresh = fast_real(mislabeled_rate_max_thresh)
                self.mean_error_max_thresh = fast_real(mean_error_max_thresh) 
                self.sample_size_of_losses = fast_real(sample_size_of_losses)
            
            @staticmethod
            def build(disabled, prop_fit_config):
                disabled = (disabled or prop_fit_config['disabled']) if 'disabled' in prop_fit_config else disabled
                single_batch_size = prop_fit_config['single_batch_size']
                total_iters = prop_fit_config['total_iters']
                abs_loss_min_thresh = prop_fit_config['abs_loss_min_thresh']
                loss_min_thresh = prop_fit_config['loss_min_thresh']
                smoothness_max_thresh = prop_fit_config['smoothness_max_thresh']
                mislabeled_rate_max_thresh = prop_fit_config['mislabeled_rate_max_thresh']
                mean_error_max_thresh = prop_fit_config['mean_error_max_thresh']
                sample_size_of_losses = prop_fit_config['sample_size_of_losses']
                return Config.PreCheck.Proper_Fitting(disabled, single_batch_size, total_iters, abs_loss_min_thresh, loss_min_thresh, smoothness_max_thresh, mislabeled_rate_max_thresh, mean_error_max_thresh, sample_size_of_losses)

        class Instance_wise_Operation:
            def __init__(self, disabled, sample_size, trials):
                self.disabled = disabled
                self.sample_size = fast_real(sample_size)
                self.trials = fast_real(trials)

            @staticmethod
            def build(disabled, inst_wise_op_config):
                disabled = (disabled or inst_wise_op_config['disabled']) if 'disabled' in inst_wise_op_config else disabled
                return Config.PreCheck.Instance_wise_Operation(disabled, inst_wise_op_config['sample_size'],inst_wise_op_config['trials'])

    class OverfitCheck:

        def __init__(self, start, period, regr_perf_thresh, classif_perf_thresh, patience, fail_on, disabled, act, weight, bias, grad, loss):
            self.start = fast_real(start)
            self.period = fast_real(period)
            self.regr_perf_thresh = fast_real(regr_perf_thresh)
            self.classif_perf_thresh = fast_real(classif_perf_thresh)
            self.patience = fast_real(patience)
            self.fail_on = fail_on
            self.disabled = disabled
            self.act = act
            self.weight = weight
            self.bias = bias
            self.grad = grad
            self.loss = loss

        @staticmethod
        def build(overfit_check_config):
            start, period, patience = overfit_check_config['start'], overfit_check_config['period'], overfit_check_config['patience']
            regr_perf_thresh, classif_perf_thresh = overfit_check_config['regr_perf_thresh'], overfit_check_config['classif_perf_thresh']
            fail_on, disabled = overfit_check_config['fail_on'], overfit_check_config['disabled']
            act = Config.OverfitCheck.Activation.build(start, period, disabled, overfit_check_config['Activation'])
            weight = Config.OverfitCheck.Weight.build(start, period, disabled, overfit_check_config['Weight'])
            bias = Config.OverfitCheck.Bias.build(start, period, disabled, overfit_check_config['Bias'])
            grad = Config.OverfitCheck.Gradient.build(start, period, disabled, overfit_check_config['Gradient'])
            loss = Config.OverfitCheck.Loss.build(start, period, disabled, overfit_check_config['Loss'])
            return Config.OverfitCheck(start, period, regr_perf_thresh, classif_perf_thresh, patience, fail_on, disabled, act, weight, bias, grad, loss)

        class Activation:
            
            def __init__(self, start, period, disabled, dead, sat, dist, out, range, numeric_ins):
                self.start = fast_real(start)
                self.period = fast_real(period)
                self.disabled = disabled
                self.dead = dead
                self.sat = sat
                self.dist = dist
                self.out = out
                self.range = range
                self.numeric_ins = numeric_ins

            @staticmethod
            def build(start, period, disabled, act_config):
                start = act_config['start'] if 'start' in act_config else start
                period = act_config['period']  if 'period' in act_config else period
                disabled = (disabled or act_config['disabled']) if 'disabled' in act_config else disabled
                dead = Config.OverfitCheck.Activation.Dead.build(disabled, act_config['Dead'])
                sat = Config.OverfitCheck.Activation.Saturation.build(disabled, act_config['Saturation'])
                dist = Config.OverfitCheck.Activation.Distribution.build(disabled, act_config['Distribution'])
                out = Config.OverfitCheck.Activation.Output.build(disabled, act_config['Output'])
                range = Config.OverfitCheck.Activation.Range.build(disabled, act_config['Range'])
                numeric_ins = Config.OverfitCheck.Activation.Numerical_Instability.build(disabled, act_config['Numerical_Instability'])
                return Config.OverfitCheck.Activation(start, period, disabled, dead, sat, dist, out, range, numeric_ins)

            class Numerical_Instability:
                def __init__(self, disabled):
                    self.disabled = disabled

                @staticmethod
                def build(disabled, numeric_ins_config):
                    disabled = (disabled or numeric_ins_config['disabled']) if 'disabled' in numeric_ins_config else disabled
                    return Config.OverfitCheck.Weight.Numerical_Instability(disabled)
                    
            class Output:
                def __init__(self, disabled, patience):
                    self.disabled = disabled
                    self.patience = patience

                @staticmethod
                def build(disabled, out_config):
                    disabled = (disabled or out_config['disabled']) if 'disabled' in out_config else disabled
                    patience = out_config['patience']
                    return Config.OverfitCheck.Activation.Output(disabled, patience)
            
            class Range:
                def __init__(self, disabled):
                    self.disabled = disabled

                @staticmethod
                def build(disabled, range_config):
                    disabled = (disabled or range_config['disabled']) if 'disabled' in range_config else disabled
                    return Config.OverfitCheck.Activation.Range(disabled)
            
            class Dead:
                def __init__(self, disabled, act_min_thresh, act_maj_percentile, neurons_ratio_max_thresh):
                    self.disabled = disabled
                    self.act_min_thresh = fast_real(act_min_thresh)
                    self.act_maj_percentile = fast_real(act_maj_percentile)
                    self.neurons_ratio_max_thresh = fast_real(neurons_ratio_max_thresh)
                @staticmethod
                def build(disabled, dead_config):
                    disabled = (disabled or dead_config['disabled']) if 'disabled' in dead_config else disabled
                    return Config.OverfitCheck.Activation.Dead(disabled, dead_config['act_min_thresh'], dead_config['act_maj_percentile'], dead_config['neurons_ratio_max_thresh'])

            class Saturation:
                def __init__(self, disabled, ro_histo_bins_count, ro_histo_min, ro_histo_max, ro_max_thresh, neurons_ratio_max_thresh):
                    self.disabled = disabled
                    self.ro_histo_bins_count = fast_real(ro_histo_bins_count)
                    self.ro_histo_min = fast_real(ro_histo_min)
                    self.ro_histo_max = fast_real(ro_histo_max)
                    self.ro_max_thresh = fast_real(ro_max_thresh)
                    self.neurons_ratio_max_thresh = fast_real(neurons_ratio_max_thresh)
                @staticmethod
                def build(disabled, sat_config):
                    disabled = (disabled or sat_config['disabled']) if 'disabled' in sat_config else disabled
                    ro_histo_bins_count = sat_config['ro_histo_bins_count']
                    ro_histo_min = sat_config['ro_histo_min']
                    ro_histo_max = sat_config['ro_histo_max']
                    ro_max_thresh = sat_config['ro_max_thresh']
                    neurons_ratio_max_thresh = sat_config['neurons_ratio_max_thresh']
                    return Config.OverfitCheck.Activation.Saturation(disabled, ro_histo_bins_count, ro_histo_min, ro_histo_max, ro_max_thresh, neurons_ratio_max_thresh)

            class Distribution:
                def __init__(self, disabled, std_acts_min_thresh, std_acts_max_thresh, f_test_alpha):
                    self.disabled = disabled
                    self.std_acts_min_thresh = std_acts_min_thresh
                    self.std_acts_max_thresh = std_acts_max_thresh
                    self.f_test_alpha = f_test_alpha
                @staticmethod
                def build(disabled, dist_config):
                    disabled = (disabled or dist_config['disabled']) if 'disabled' in dist_config else disabled
                    return Config.OverfitCheck.Activation.Distribution(disabled, dist_config['std_acts_min_thresh'], dist_config['std_acts_max_thresh'], dist_config['f_test_alpha'])

        class Weight:
            
            def __init__(self, start, period, disabled, dead, neg, div, numeric_ins):
                self.start = fast_real(start)
                self.period = fast_real(period)
                self.disabled = disabled
                self.dead = dead
                self.neg = neg
                self.div = div
                self.numeric_ins = numeric_ins

            @staticmethod
            def build(start, period, disabled, weight_config):
                start = weight_config['start'] if 'start' in weight_config else start
                period = weight_config['period']  if 'period' in weight_config else period
                disabled = (disabled or weight_config['disabled']) if 'disabled' in weight_config else disabled
                dead = Config.OverfitCheck.Weight.Dead.build(disabled, weight_config['Dead'])
                neg = Config.OverfitCheck.Weight.Negative.build(disabled, weight_config['Negative'])
                div = Config.OverfitCheck.Weight.Diverging.build(disabled, weight_config['Diverging'])
                numeric_ins = Config.OverfitCheck.Weight.Numerical_Instability.build(disabled, weight_config['Numerical_Instability'])
                return Config.OverfitCheck.Weight(start, period, disabled, dead, neg, div, numeric_ins)

            class Numerical_Instability:
                def __init__(self, disabled):
                    self.disabled = disabled

                @staticmethod
                def build(disabled, numeric_ins_config):
                    disabled = (disabled or numeric_ins_config['disabled']) if 'disabled' in numeric_ins_config else disabled
                    return Config.OverfitCheck.Weight.Numerical_Instability(disabled)

            class Dead:
                def __init__(self, disabled, value_min_thresh, ratio_max_thresh):
                    self.disabled = disabled
                    self.value_min_thresh = value_min_thresh
                    self.ratio_max_thresh = ratio_max_thresh
                    
                @staticmethod
                def build(disabled, dead_config):
                    disabled = (disabled or dead_config['disabled']) if 'disabled' in dead_config else disabled
                    return Config.OverfitCheck.Weight.Dead(disabled, dead_config['value_min_thresh'], dead_config['ratio_max_thresh'])

            class Negative:
                def __init__(self, disabled, ratio_max_thresh):
                    self.disabled = disabled
                    self.ratio_max_thresh = ratio_max_thresh
                
                @staticmethod
                def build(disabled, neg_config):
                    disabled = (disabled or neg_config['disabled']) if 'disabled' in neg_config else disabled
                    return Config.OverfitCheck.Weight.Negative(disabled, neg_config['ratio_max_thresh'])

            class Diverging:
                def __init__(self, disabled, window_size, mav_max_thresh, inc_rate_max_thresh):
                    self.disabled = disabled
                    self.window_size = window_size
                    self.mav_max_thresh = mav_max_thresh
                    self.inc_rate_max_thresh = inc_rate_max_thresh
                
                @staticmethod
                def build(disabled, div_config):
                    disabled = (disabled or div_config['disabled']) if 'disabled' in div_config else disabled
                    return Config.OverfitCheck.Weight.Diverging(disabled, div_config['window_size'], div_config['mav_max_thresh'], div_config['inc_rate_max_thresh'])

        class Bias:
            
            def __init__(self, start, period, disabled, div, numeric_ins):
                self.start = fast_real(start)
                self.period = fast_real(period)
                self.disabled = disabled
                self.div = div
                self.numeric_ins = numeric_ins

            @staticmethod
            def build(start, period, disabled, bias_config):
                start = bias_config['start'] if 'start' in bias_config else start
                period = bias_config['period']  if 'period' in bias_config else period
                disabled = (disabled or bias_config['disabled']) if 'disabled' in bias_config else disabled
                div = Config.OverfitCheck.Bias.Diverging.build(disabled, bias_config['Diverging'])
                numeric_ins = Config.OverfitCheck.Bias.Numerical_Instability.build(disabled, bias_config['Numerical_Instability'])
                return Config.OverfitCheck.Bias(start, period, disabled, div, numeric_ins)

            class Numerical_Instability:
                def __init__(self, disabled):
                    self.disabled = disabled

                @staticmethod
                def build(disabled, numeric_ins_config):
                    disabled = (disabled or numeric_ins_config['disabled']) if 'disabled' in numeric_ins_config else disabled
                    return Config.OverfitCheck.Bias.Numerical_Instability(disabled)

            class Diverging:
                def __init__(self, disabled, window_size, mav_max_thresh, inc_rate_max_thresh):
                    self.disabled = disabled
                    self.window_size = window_size
                    self.mav_max_thresh = mav_max_thresh
                    self.inc_rate_max_thresh = inc_rate_max_thresh
                
                @staticmethod
                def build(disabled, div_config):
                    disabled = (disabled or div_config['disabled']) if 'disabled' in div_config else disabled
                    return Config.OverfitCheck.Bias.Diverging(disabled, div_config['window_size'], div_config['mav_max_thresh'], div_config['inc_rate_max_thresh'])

        class Gradient:

            def __init__(self, start, period, disabled, vanish, explod, unstab, numeric_ins):
                self.start = fast_real(start)
                self.period = fast_real(period)
                self.disabled = disabled
                self.vanish = vanish
                self.explod = explod
                self.unstab = unstab
                self.numeric_ins = numeric_ins

            @staticmethod
            def build(start, period, disabled, grad_config):
                start = grad_config['start'] if 'start' in grad_config else start
                period = grad_config['period']  if 'period' in grad_config else period
                disabled = (disabled or grad_config['disabled']) if 'disabled' in grad_config else disabled
                vanish = Config.OverfitCheck.Gradient.Vanishing.build(disabled, grad_config['Vanishing'])
                explod = Config.OverfitCheck.Gradient.Exploding.build(disabled, grad_config['Exploding'])
                unstab = Config.OverfitCheck.Gradient.Unstable_Learning.build(disabled, grad_config['Unstable_Learning'])
                numeric_ins = Config.OverfitCheck.Gradient.Numerical_Instability.build(disabled, grad_config['Numerical_Instability'])
                return Config.OverfitCheck.Gradient(start, period, disabled, vanish, explod, unstab, numeric_ins)
            
            class Numerical_Instability:
                def __init__(self, disabled):
                    self.disabled = disabled

                @staticmethod
                def build(disabled, numeric_ins_config):
                    disabled = (disabled or numeric_ins_config['disabled']) if 'disabled' in numeric_ins_config else disabled
                    return Config.OverfitCheck.Gradient.Numerical_Instability(disabled)

            class Vanishing:
                def __init__(self, disabled, window_size, mav_min_thresh, dec_rate_min_thresh):
                    self.disabled = disabled
                    self.window_size = window_size
                    self.mav_min_thresh = mav_min_thresh
                    self.dec_rate_min_thresh = dec_rate_min_thresh

                @staticmethod
                def build(disabled, van_config):
                    disabled = (disabled or van_config['disabled']) if 'disabled' in van_config else disabled
                    return Config.OverfitCheck.Gradient.Vanishing(disabled, van_config['window_size'], van_config['mav_min_thresh'], van_config['dec_rate_min_thresh'])

            class Exploding:
                def __init__(self, disabled, window_size, mav_max_thresh, inc_rate_max_thresh):
                    self.disabled = disabled
                    self.window_size = window_size
                    self.mav_max_thresh = mav_max_thresh
                    self.inc_rate_max_thresh = inc_rate_max_thresh

                @staticmethod
                def build(disabled, exp_config):
                    disabled = (disabled or exp_config['disabled']) if 'disabled' in exp_config else disabled
                    return Config.OverfitCheck.Gradient.Exploding(disabled, exp_config['window_size'], exp_config['mav_max_thresh'], exp_config['inc_rate_max_thresh'])

            class Unstable_Learning:
                def __init__(self, disabled, high_updates_max_thresh, low_updates_min_thresh):
                    self.disabled = disabled
                    self.high_updates_max_thresh = high_updates_max_thresh
                    self.low_updates_min_thresh = low_updates_min_thresh

                @staticmethod
                def build(disabled, uns_config):
                    disabled = (disabled or uns_config['disabled']) if 'disabled' in uns_config else disabled
                    return Config.OverfitCheck.Gradient.Unstable_Learning(disabled, uns_config['high_updates_max_thresh'], uns_config['low_updates_min_thresh'])

        class Loss:

            def __init__(self, start, period, disabled, non_dec, fluct, div, rep, over_reg, numeric_ins):
                self.start = fast_real(start)
                self.period = fast_real(period)
                self.disabled = disabled
                self.non_dec = non_dec
                self.fluct = fluct
                self.div = div
                self.rep = rep
                self.over_reg = over_reg
                self.numeric_ins = numeric_ins

            @staticmethod
            def build(start, period, disabled, loss_config):
                start = loss_config['start'] if 'start' in loss_config else start
                period = loss_config['period']  if 'period' in loss_config else period
                disabled = (disabled or loss_config['disabled']) if 'disabled' in loss_config else disabled
                non_dec = Config.OverfitCheck.Loss.NonDecreasing.build(disabled, loss_config['NonDecreasing'])
                fluct = Config.OverfitCheck.Loss.Fluctuating.build(disabled, loss_config['Fluctuating'])
                div = Config.OverfitCheck.Loss.Diverging.build(disabled, loss_config['Diverging'])
                rep = Config.OverfitCheck.Loss.Representativeness.build(disabled, loss_config['Representativeness'])
                over_reg = Config.OverfitCheck.Loss.Overwhelming_Reg.build(disabled, loss_config['Overwhelming_Reg'])
                numeric_ins = Config.OverfitCheck.Loss.Numerical_Instability.build(disabled, loss_config['Numerical_Instability'])
                return Config.OverfitCheck.Loss(start, period, disabled, non_dec, fluct, div, rep, over_reg, numeric_ins)

            class Numerical_Instability:
                def __init__(self, disabled):
                    self.disabled = disabled

                @staticmethod
                def build(disabled, numeric_ins_config):
                    disabled = (disabled or numeric_ins_config['disabled']) if 'disabled' in numeric_ins_config else disabled
                    return Config.OverfitCheck.Loss.Numerical_Instability(disabled)

            class NonDecreasing:
                def __init__(self, disabled, window_size, decr_percentage):
                    self.disabled = disabled
                    self.window_size = window_size
                    self.decr_percentage = decr_percentage
                
                @staticmethod
                def build(disabled, non_dec_config):
                    disabled = (disabled or non_dec_config['disabled']) if 'disabled' in non_dec_config else disabled
                    return Config.OverfitCheck.Loss.NonDecreasing(disabled, non_dec_config['window_size'], non_dec_config['decr_percentage'])
            
            class Diverging:
                def __init__(self, disabled, window_size, incr_abs_rate_max_thresh):
                    self.disabled = disabled
                    self.window_size = window_size
                    self.incr_abs_rate_max_thresh = incr_abs_rate_max_thresh
                
                @staticmethod
                def build(disabled, div_config):
                    disabled = (disabled or div_config['disabled']) if 'disabled' in div_config else disabled
                    return Config.OverfitCheck.Loss.Diverging(disabled, div_config['window_size'], div_config['incr_abs_rate_max_thresh'])
            
            class Fluctuating:
                def __init__(self, disabled, window_size, smoothness_ratio_min_thresh):
                    self.disabled = disabled
                    self.window_size = window_size
                    self.smoothness_ratio_min_thresh = smoothness_ratio_min_thresh

                @staticmethod
                def build(disabled, fluct_config):
                    disabled = (disabled or fluct_config['disabled']) if 'disabled' in fluct_config else disabled
                    return Config.OverfitCheck.Loss.Fluctuating(disabled, fluct_config['window_size'], fluct_config['smoothness_ratio_min_thresh'])
            
            class Representativeness:
                def __init__(self, disabled, abs_corr_min_thresh):
                    self.disabled = disabled
                    self.abs_corr_min_thresh = abs_corr_min_thresh

                @staticmethod
                def build(disabled, rep_config):
                    disabled = (disabled or rep_config['disabled']) if 'disabled' in rep_config else disabled
                    return Config.OverfitCheck.Loss.Representativeness(disabled, rep_config['abs_corr_min_thresh'])

            class Overwhelming_Reg:
                def __init__(self, disabled, window_size, growth_rate_max_thresh):
                    self.disabled = disabled
                    self.window_size = window_size
                    self.growth_rate_max_thresh = growth_rate_max_thresh

                @staticmethod
                def build(disabled, over_reg_config):
                    disabled = (disabled or over_reg_config['disabled']) if 'disabled' in over_reg_config else disabled
                    return Config.OverfitCheck.Loss.Overwhelming_Reg(disabled, over_reg_config['window_size'], over_reg_config['growth_rate_max_thresh'])

    class PostCheck:

        def __init__(self, start, period, fail_on, disabled, switch_mode_consist, corrup_lbls, data_augm):
            self.start = fast_real(start)
            self.period = fast_real(period)
            self.fail_on = fail_on
            self.disabled = disabled
            self.switch_mode_consist = switch_mode_consist
            self.corrup_lbls = corrup_lbls
            self.data_augm = data_augm

        @staticmethod
        def build(post_check_config):
            start, period = post_check_config['start'], post_check_config['period']
            fail_on, disabled = post_check_config['fail_on'], post_check_config['disabled']
            switch_mode_consist = Config.PostCheck.Switch_Mode_Consistency.build(start, period, disabled, post_check_config['Switch_Mode_Consistency'])
            corrup_lbls = Config.PostCheck.Corrupted_Labels.build(disabled, post_check_config['Corrupted_Labels'])
            data_augm = Config.PostCheck.Data_Augm.build(disabled, post_check_config['Data_Augm'])
            return Config.PostCheck(start, period, fail_on, disabled, switch_mode_consist, corrup_lbls, data_augm)
        
        class Corrupted_Labels:

            def __init__(self, disabled, batch_size, warmup_epochs, total_epochs, patience, perf_improv_ratio_min_thresh):
                self.disabled = disabled
                self.batch_size = fast_real(batch_size)
                self.warmup_epochs = fast_real(warmup_epochs)
                self.total_epochs = fast_real(total_epochs)
                self.patience = fast_real(patience)
                self.perf_improv_ratio_min_thresh = fast_real(perf_improv_ratio_min_thresh)
            
            @staticmethod
            def build(disabled, corrup_lbls_config):
                disabled = (disabled or corrup_lbls_config['disabled']) if 'disabled' in corrup_lbls_config else disabled
                batch_size = corrup_lbls_config['batch_size']
                warmup_epochs = corrup_lbls_config['warmup_epochs']
                total_epochs = corrup_lbls_config['total_epochs']
                patience = corrup_lbls_config['patience']
                perf_improv_ratio_min_thresh = corrup_lbls_config['perf_improv_ratio_min_thresh']
                return Config.PostCheck.Corrupted_Labels(disabled, batch_size, warmup_epochs, total_epochs, patience, perf_improv_ratio_min_thresh)

        class Data_Augm:

            def __init__(self, disabled, batch_size, total_epochs, valid_sample_size, sim_with_augm_min_thresh):
                self.disabled = disabled
                self.batch_size = fast_real(batch_size)
                self.total_epochs = fast_real(total_epochs)
                self.valid_sample_size = fast_real(valid_sample_size)
                self.sim_with_augm_min_thresh = fast_real(sim_with_augm_min_thresh)
            
            @staticmethod
            def build(disabled, data_augm_config):
                disabled = (disabled or data_augm_config['disabled']) if 'disabled' in data_augm_config else disabled
                batch_size = data_augm_config['batch_size']
                total_epochs = data_augm_config['total_epochs']
                valid_sample_size = data_augm_config['valid_sample_size']
                sim_with_augm_min_thresh = data_augm_config['sim_with_augm_min_thresh']
                return Config.PostCheck.Data_Augm(disabled, batch_size, total_epochs, valid_sample_size, sim_with_augm_min_thresh)

        class Switch_Mode_Consistency:

            def __init__(self, start, period, disabled, batch_size, total_epochs, valid_sample_size, sim_after_switch_mode_min_thresh, relative_loss_diff_max_thresh):
                self.start = start 
                self.period = period 
                self.disabled = disabled
                self.batch_size = fast_real(batch_size)
                self.total_epochs = fast_real(total_epochs)
                self.valid_sample_size = fast_real(valid_sample_size)
                self.sim_after_switch_mode_min_thresh = fast_real(sim_after_switch_mode_min_thresh)
                self.relative_loss_diff_max_thresh = fast_real(relative_loss_diff_max_thresh)

            @staticmethod
            def build(start, period, disabled, switch_mode_cons_config):
                start = switch_mode_cons_config['start'] if 'start' in switch_mode_cons_config else start
                period = switch_mode_cons_config['period']  if 'period' in switch_mode_cons_config else period
                disabled = (disabled or switch_mode_cons_config['disabled']) if 'disabled' in switch_mode_cons_config else disabled
                batch_size = switch_mode_cons_config['batch_size']
                total_epochs = switch_mode_cons_config['total_epochs']
                valid_sample_size = switch_mode_cons_config['valid_sample_size']
                sim_after_switch_mode_min_thresh = switch_mode_cons_config['sim_after_switch_mode_min_thresh']
                relative_loss_diff_max_thresh = switch_mode_cons_config['relative_loss_diff_max_thresh']
                return Config.PostCheck.Switch_Mode_Consistency(start, period, disabled, batch_size, total_epochs, valid_sample_size, sim_after_switch_mode_min_thresh, relative_loss_diff_max_thresh)

#Logging
def build_log_file_path(app_path, app_name):
    log_dir_path = os.path.join(app_path, LOG_DIR)
    if not(os.path.exists(log_dir_path)):
        os.makedirs(log_dir_path)
    return os.path.join(log_dir_path, f'{app_name}.log')

def file_logger(file_path, app_name):
    logger = logging.getLogger(f'TheDeepChecker: {app_name} Logs')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(file_path)
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger

def console_logger(app_name):
    logger = logging.getLogger(f'TheDeepChecker: {app_name} Logs')
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger




