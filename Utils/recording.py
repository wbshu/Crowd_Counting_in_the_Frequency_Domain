'''
@author: weiboshu2

@file: recording.py

@time: 2021/9/5 22:02

@purpose: take charge in all kinds of observation & recording activities in the training process.

'''

import numpy as np
import torch

from Utils.metric_measure import Average_Metric, Time_Metric
import logging


class Logger():
    def log(self, results: dict):
        logging.info(str(results))


class Execution_Condition():
    '''
        This kind of class is specific to the 'execution condition' of diverse behavious in the observation & recording.
    '''

    def __init__(self, recorder):
        self.recorder = recorder

    def judge(self):
        raise NotImplementedError


class EC_compose_condition(Execution_Condition):
    def __init__(self, recorder, condition_list):
        super(EC_compose_condition, self).__init__(recorder)
        self.conditions = condition_list

    def judge(self):
        for i in self.conditions:
            if i.judge():
                continue
            else:
                return False
        return True


class EC_epoch_exact_division(Execution_Condition):
    def __init__(self, recorder, divisor):
        super(EC_epoch_exact_division, self).__init__(recorder)
        self.divisor = divisor

    def judge(self):
        if self.recorder.epoch % self.divisor == 0:
            return True
        else:
            return False


class EC_epoch_lower_bound(Execution_Condition):
    def __init__(self, recorder, lower_bound):
        super(EC_epoch_lower_bound, self).__init__(recorder)
        self.threshold = lower_bound

    def judge(self):
        if self.recorder.epoch >= self.threshold:
            return True
        else:
            return False


class EC_set_match_prefix(Execution_Condition):
    def __init__(self, recorder, dealing_prefix):
        super(EC_set_match_prefix, self).__init__(recorder)
        self.dealing_prefix = dealing_prefix

    def judge(self):
        if type(self.dealing_prefix) is list or type(self.dealing_prefix) is tuple:
            for i in self.dealing_prefix:
                if self.recorder.current_set.startswith(i):
                    return True
        elif type(self.dealing_prefix) is str:
            if self.recorder.current_set.startswith(self.dealing_prefix):
                return True
        else:
            raise NotImplementedError
        return False


class Data_filtering():
    '''
    This kind of class is responsible for filtering & preprocessing directly collected data from training process.
    '''

    def __init__(self, recorder, conditions=None):
        self.recorder = recorder
        self.conditions = conditions

    def filtering(self, name, var):
        raise NotImplementedError


class Record_Operation():
    '''
    This kind of class is used for diverse recording behaviours. ERO means recording operations after each epoch.
    BRO means recording operations after each batch. we offer some basic operations in the file, you can implement
    your desired operations by extends the Basic class here.
    '''

    def __init__(self, recorder, conditions=None):
        self.recorder = recorder
        self.conditions = conditions

    def record_operate(self):
        raise NotImplementedError


class ERO_record_mae_mse(Record_Operation):
    def record_operate(self):
        if self.conditions is None or self.conditions.judge():
            name_1 = self.recorder.current_set + '_mae'
            self.recorder.results[name_1] = self.recorder.recorders['mae'].get_avg()
            name_2 = self.recorder.current_set + '_mse'
            self.recorder.results[name_2] = np.sqrt(self.recorder.recorders['mse'].get_avg())
            return self.recorder.results[name_1], self.recorder.results[name_2]


class ERO_record_loss(Record_Operation):
    def record_operate(self):
        if self.conditions is None or self.conditions.judge():
            name = self.recorder.current_set + '_loss'
            self.recorder.results[name] = self.recorder.recorders['loss'].get_avg()
            return self.recorder.results[name]


class ERO_record_time(Record_Operation):
    def record_operate(self):
        logging.info('-' * 5 + 'Epoch {}'.format(self.recorder.epoch) + '-' * 5)
        if self.conditions is None or self.conditions.judge():
            logging.info('Epoch {} {},  Cost {:.1f} sec'
                         .format(self.recorder.epoch, self.recorder.current_set,
                                 self.recorder.recorders['time'].get_time_consumption()))


class BRO_record_mae_mse(Record_Operation):
    def record_operate(self):
        if self.conditions is None or self.conditions.judge():
            with torch.no_grad():
                N = self.recorder.collected_data[self.recorder.current_set]['input'][0].size(0)
                res = torch.sum(self.recorder.collected_data[self.recorder.current_set]['output'][0].view(N, -1),
                                dim=-1) - self.recorder.collected_data[self.recorder.current_set]['people_count'][0]
                res = res.cpu().numpy()
                self.recorder.recorders['mae'].update(np.mean(abs(res)), N)
                self.recorder.recorders['mse'].update(np.mean(res * res), N)


class BRO_record_loss(Record_Operation):
    def record_operate(self):
        if self.conditions is None or self.conditions.judge():
            with torch.no_grad():
                N = self.recorder.collected_data[self.recorder.current_set]['input'][0].size(0)
                self.recorder.recorders['loss'].update(
                    self.recorder.collected_data[self.recorder.current_set]['loss'][0].item(), N)


class Recorder():
    '''
        this class works for recording results from training process as well as recording best model parameters in training process
        The recorder class basically have two basic elements:
            Data structure: we keep two dictionaries for each set (train/val/test/...), the dictionary records diverse data
                            the first dictionary keeps the direct data ; the second dictionary keeps the indirect data.
                            all the first dictionaries are organized as the dictionary `self.collected_data', and they
                            are sorted according to their corresponding set (train/val/test/...). Similarly, all the second
                            dictionaries are organized as the dictionary `self.results', and they are also sorted
                            according to their corresponding set (train/val/test/...).
            Operations: the operations includes all kinds of observation & recording manipulations (e.g., the behaviour of
                        recording mae/mse). We use the thought of AOP (Aspect Oriented Programming) here to decouple the
                        observation & recording behaviour from the mainstream of the training process. If you want to
                        make some bespoke operations, just extends the `Record_Operation' class and insert it into proper
                        position in the training flow.
                        We sort the operations into 3 categories: filtering the direct data ; operations after training
                                                                 batch; operations after training epoch

        If you just want to use the code, encapsulating the observation & recording operations into the training process
        is even simpler than our decoupling design here. But if you want to begin your own research journey based on the
        codes, and you have been familiar with the simple decoupling structure here, you will be gradually aware of the
        advantage of using the scalable code design here to decouple the observation & recording operations from the mainstream
        of the training.
    '''

    def __init__(self):
        # basic data structure
        self.results = {}  # record all the indirect data
        self.recorders = {}  # recorders used for recoding all kinds of data
        self.collected_data = {}  # record all the direct data coming from the training process

        # operataions list
        self.filters_after_data = []  # filters for filtering the direct data
        self.operations_after_epoch = []  # operations after each epoch
        self.operations_after_batch = []  # operations after each batch

        # attributes deciding the current state of the recorder
        self.epoch = 0
        self.current_set = None

    '''
        basic methods
    '''

    def set_state(self, set_name, epoch):
        if set_name not in self.collected_data.keys():
            self.collected_data[set_name] = {}
        self.current_set = set_name
        self.epoch = epoch

    def reset_records(self):
        self.results = {}
        self.collected_data = {}

    def reset_states(self):
        self.current_set = None
        self.epoch = 0

    def reset_recorders(self):
        for i in list(self.recorders.keys()):
            self.recorders[i].reset()

    def register(self, name, variable):
        '''
         note: this method works for registering variables used for generating recorded data, it will also take charge
              in simple data clean.  Since it will return the data for outside using after registration, please don't
              do complex processing to the input data.  Or you can do it, but please return outside
               the data same to the input except for some necessary aspects such as 'setting retain_grad=True for observing
               derivative.

         the registered common used name:
             input, target, output, loss, optimizer, people_count, count_coefficient
        Returns:

        '''
        if type(variable) is tuple:
            self.collected_data[self.current_set][name] = list(variable)
        else:
            self.collected_data[self.current_set][name] = [variable, ]

        self.filtering(name, self.collected_data[self.current_set][name])

        return self.collected_data[self.current_set][name][0]

    def filtering(self, name, var):
        for i in self.filters_after_data:
            name, var = i.filtering(name, var)

    def record_for_batch(self):
        for i in self.operations_after_batch:
            i.record_operate()

    def get_records(self):
        out = []
        for i in self.operations_after_epoch:
            result = i.record_operate()
            if result is not None:
                if type(result) is tuple:
                    result = list(result)
                else:
                    result = [result, ]
                out.extend(result)
        return tuple(out)

    def record(self, logger):
        logger.log(self.results)

    '''
    high level user-friendly methods
    '''

    def basic_setting(self):
        '''
         record train_ set's mae&mse&loss and val_ set's mae&mse
        '''

        # register recorders
        self.recorders['mae'] = Average_Metric()
        self.recorders['mse'] = Average_Metric()
        self.recorders['loss'] = Average_Metric()
        self.recorders['time'] = Time_Metric()

        # register operations,  their orders here decide their execution orders after each batch/epoch.
        self.operations_after_epoch.extend(
            [ERO_record_mae_mse(self), ERO_record_loss(self, conditions=EC_set_match_prefix(self, 'train')),
             ERO_record_time(self)])
        self.operations_after_batch.extend(
            [BRO_record_mae_mse(self), BRO_record_loss(self, conditions=EC_set_match_prefix(self, 'train'))])



