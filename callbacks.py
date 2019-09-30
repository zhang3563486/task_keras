import tqdm
import numpy as np
import csv
import six
from collections import OrderedDict
from collections import Iterable
from sklearn.metrics import roc_curve, auc

from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.callbacks import CSVLogger
from keras.callbacks import LearningRateScheduler

class Evaluate(Callback):
    def __init__(
        self,
        generator,
        eval_steps,
        verbose=1):

        super(Evaluate, self).__init__()
        self.generator = generator
        self.eval_steps = eval_steps
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs=None):
        if epoch == 0 or (epoch+1) % 5 == 0:
            logs = logs or {}
            self.auc_score = self.evaluate()

            if self.verbose == 1:
                print('Epoch {:4d} - AUC : {:.4f}'.format(epoch+1, self.auc_score))
            
        logs['eval_auc'] = self.auc_score

    def evaluate(self):
        label = []
        pred = []
        for i in tqdm.trange(self.eval_steps):
            g = next(self.generator)
            result = self.model.predict_on_batch(g[0])
            label.append(g[1][0,1])
            pred.append(result[0,1])
        
        fpr, tpr, thresholds = roc_curve(label, pred, pos_label=1)
        auc_score = auc(fpr, tpr)
        return auc_score


class CustomBatchLogger(CSVLogger):
    def __init__(self, filename, separator=',', append=False):
        super().__init__(filename, separator, append)

    def on_train_begin(self, logs=None):
        super().on_train_begin(logs)

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}

        def handle_value(k):
            is_zero_dim_ndarray = isinstance(k, np.ndarray) and k.ndim == 0
            if isinstance(k, six.string_types):
                return k
            elif isinstance(k, Iterable) and not is_zero_dim_ndarray:
                return '"[%s]"' % (', '.join(map(str, k)))
            else:
                return k

        if self.keys is None:
            self.keys = sorted(logs.keys())

        if self.model.stop_training:
            # We set NA so that csv parsers do not fail for this last batch.
            logs = dict([(k, logs[k] if k in logs else 'NA') for k in self.keys])

        if not self.writer:
            class CustomDialect(csv.excel):
                delimiter = self.sep
            fieldnames = ['batch'] + self.keys
            if six.PY2:
                fieldnames = [unicode(x) for x in fieldnames]
            self.writer = csv.DictWriter(self.csv_file,
                                         fieldnames=fieldnames,
                                         dialect=CustomDialect)
            if self.append_header:
                self.writer.writeheader()

        row_dict = OrderedDict({'batch': batch})
        row_dict.update((key, handle_value(logs[key])) for key in self.keys)
        self.writer.writerow(row_dict)
        self.csv_file.flush()

    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_train_end(self, logs=None):
        super().on_train_end(logs)


class LearningRateBatchScheduler(LearningRateScheduler):
    def __init__(self, schedule):
        super(LearningRateBatchScheduler, self).__init__()
        self.schedule = schedule
        
    def on_batch_begin(self, batch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        lr = float(K.get_value(self.model.optimizer.lr))
        try:  # new API
            lr = self.schedule(epoch, lr)
        except TypeError:  # old API for backward compatibility
            lr = self.schedule(epoch)
        if not isinstance(lr, (float, np.float32, np.float64)):
            raise ValueError('The output of the "schedule" function '
                             'should be float.')
        K.set_value(self.model.optimizer.lr, lr)

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)


def callback_checkpoint(filepath, monitor, verbose, mode, save_best_only, save_weights_only):
    return ModelCheckpoint(filepath=filepath,
                           monitor=monitor,
                           verbose=verbose,
                           mode=mode,
                           save_best_only=save_best_only,
                           save_weights_only=save_weights_only)

def callback_epochlogger(filename, separator, append):
    return CSVLogger(filename=filename,
                     separator=separator,
                     append=append)

def callback_batchlogger(filename, separator, append):
    return CustomBatchLogger(filename=filename,
                             separator=separator,
                             append=append)

def callback_tensorboard(log_dir, batch_size):
    return TensorBoard(log_dir=log_dir, batch_size=batch_size)

def main_schedule(initlr, warmup, mode, value, duration, total_epoch):
    def _cosine_anneal_schedule(epoch, total_epoch):
        return float((1 + ((np.pi * (epoch % total_epoch)) / total_epoch)) * initlr / 2)

    def _exponential(epoch, value, duration):
        update = epoch // duration
        return initlr * (value ** update)

    def _schedule(epoch, lr):
        if warmup and epoch < warmup:
            print('warmup!')
            return initlr*(epoch+1)/warmup

        if mode == 'constant':
            return lr
        elif mode == 'exponential':
            return _exponential(epoch-warmup, value, duration)
        elif mode == 'cosine':
            return _cosine_anneal_schedule(epoch-warmup, total_epoch-warmup)

    return _schedule

def callback_learningrate(initlr, warmup, mode, value, duration, total_epoch):
    return LearningRateScheduler(schedule=main_schedule(initlr=initlr, 
                                                        warmup=warmup,
                                                        mode=mode, 
                                                        value=value, 
                                                        duration=duration, 
                                                        total_epoch=total_epoch), verbose=1)

def callback_evaluate(generator, eval_steps):
    return Evaluate(generator=generator, eval_steps=eval_steps, verbose=1)