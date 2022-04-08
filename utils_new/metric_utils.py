"""
Class for calculating metrics for a proposed image and the original.
NOTE: all methods return per-image metrics, i.e. the number of returned values is equal to batch dimension. 
"""
import lpips
import torch
import torch.nn.functional as F
import numpy as np
from pytorch_msssim import ssim

@torch.no_grad()
def get_lpips(x_hat, x):
    """
    Calculates LPIPS(x_hat, x).
    Assumes given images are in range [0, 1].
    """
    x_hat_rescaled = (x_hat * 2.) - 1.
    x_rescaled = (x * 2.) - 1.

    loss_fn = lpips.LPIPS(net='alex').to(x_hat.device)
    lpips_loss = loss_fn.forward(x_hat_rescaled, x_rescaled)

    return lpips_loss.cpu().numpy().flatten()

@torch.no_grad()
def get_ssim(x_hat, x, range=1.):
    """
    Calculates SSIM(x_hat, x)
    """
    ssim_val = ssim(x_hat, x, data_range=range, size_average=False)

    return ssim_val.cpu().numpy().flatten()

@torch.no_grad()
def get_nmse(x_hat, x):
    """
    Calculate ||x_hat - x|| / ||x||
    """
    sse = torch.sum((x_hat - x)**2, dim=[1,2,3]) #shape [N] - sse per image
    denom = torch.sum(x**2, dim=[1,2,3]) #shape [N] - squared l2 norm per ground truth image

    nmse_val = sse / denom

    return nmse_val.cpu().numpy().flatten() 

@torch.no_grad()
def get_psnr(x_hat, x, range=1.):
    """
    Calculate 20 * log_10(range / sqrt(mse(x_hat, x))) for each image in the batch dimension.
        range is the range between high and low possible pixel values.
    """
    mse = torch.sum((x_hat - x)**2, dim=[1,2,3]) / np.prod(x_hat.shape[1:]) #shape [N]

    psnr_val = 20 * torch.log10(range / torch.sqrt(mse))

    return psnr_val.cpu().numpy().flatten() #shape [N] - per-image psnr

@torch.no_grad()
def get_sse(x_hat, x):
    """
    Calculates ||x_hat - x||^2 for each image in the batch dimension
    """
    sse_val = torch.sum((x_hat - x)**2, dim=[1,2,3])

    return sse_val.cpu().numpy().flatten() #shape [N] - sse per image

@torch.no_grad()
def get_mse(x_hat, x):
    """
    Calculates (1 / C*H*W)||x_hat - x||^2 for each image in the batch dimension
    """
    mse_val = torch.sum((x_hat - x)**2, dim=[1,2,3])

    return mse_val.cpu().numpy().flatten() / np.prod(x_hat.shape[1:])

@torch.no_grad()
def get_all_metrics(x_hat, x, range = 1.):
    """
    function for getting all image reference metrics and returning in a dict
    """
    metrics = {}

    metrics['lpips'] = get_lpips(x_hat, x)
    metrics['ssim'] = get_ssim(x_hat, x, range=range)
    metrics['nmse'] = get_nmse(x_hat, x)
    metrics['psnr'] = get_psnr(x_hat, x, range=range)
    metrics['sse'] = get_sse(x_hat, x)
    metrics['mse'] = get_mse(x_hat, x)

    return metrics

class Metrics:
    """
    A class for storing and aggregating metrics during a run.
    Metrics are stored as numpy arrays.
    """
    def __init__(self, hparams, range=1.0):
        #dicts for olding raw, image-by-image stats for each iteration.
        #e.g. self.train_metrics['iter_0']['psnr'] = [0.9, 0.1, 0.3] means that at train iteration 0, the images had psnrs of 0.9, 0.1, 0.3
        self.train_metrics = {}
        self.val_metrics = {}
        self.test_metrics = {}

        #dicts for holding summary stats for each iteration.
        #e.g. self.train_metrics_aggregate['iter_0']['mean_psnr'] = 0.5 means that training iter 0 had a mean train psnr of 0.5
        self.train_metrics_aggregate = {}
        self.val_metrics_aggregate = {}
        self.test_metrics_aggregate = {}

        #dicts for holding the best summary stats 
        #the entries have keys named after the metrics and values that are tuples of the best iteration and best value
        #e.g.  self.best_train_metrics['psnr'] = (10, 0.99) means that training psnr had its best value at iter 10, and that value is 0.99 
        self.best_train_metrics = {}
        self.best_val_metrics = {}
        self.best_test_metrics = {}

        self.range = range
        self.hparams = hparams

    def __init_iter_dict(self, cur_dict, iter_num, should_exist=False):
        """
        Helper method for initializing an iteratio metric dict if it doesn't exist
        """
        iterkey = 'iter_' + str(iter_num)

        if should_exist:
            assert iterkey in cur_dict
        elif iterkey not in cur_dict:
            cur_dict[iterkey] = {}

        return
    
    def __append_to_iter_dict(self, cur_dict, iter_metrics, iter_num):
        """
        Helper method for appending values to a given iteration metric dict 
        """
        iterkey = 'iter_' + str(iter_num)
        for key, value in iter_metrics.items():
            if key not in cur_dict[iterkey]:
                cur_dict[iterkey][key] = value
            else:
                cur_dict[iterkey][key] = np.append(cur_dict[iterkey][key], value)
        
        return
    
    def __retrieve_dict(self, iter_type, dict_type='raw'):
        """
        Helper method for validating and retrieving the correct dictionary (train, val, or test)
        """
        assert iter_type in ['train', 'val', 'test']
        assert dict_type in ['raw', 'aggregate', 'best']

        if dict_type == 'raw':
            if iter_type == 'train':
                cur_dict = self.train_metrics
            elif iter_type == 'val':
                cur_dict = self.val_metrics
            elif iter_type == 'test':
                cur_dict = self.test_metrics
        
        elif dict_type == 'aggregate':
            if iter_type == 'train':
                cur_dict = self.train_metrics_aggregate
            elif iter_type == 'val':
                cur_dict = self.val_metrics_aggregate
            elif iter_type == 'test':
                cur_dict = self.test_metrics_aggregate

        elif dict_type == 'best':
            if iter_type == 'train':
                cur_dict = self.best_train_metrics
            elif iter_type == 'val':
                cur_dict = self.best_val_metrics
            elif iter_type == 'test':
                cur_dict = self.best_test_metrics
        
        return cur_dict
    
    def get_dict(self, iter_type, dict_type='raw'):
        """Public-facing getter than calls retrieve_dict"""
        return self.__retrieve_dict(iter_type, dict_type)
    
    def get_best(self, iter_type, metric_key):
        """
        Getter method for retrieving the best iter and value for a given metric.
        If the best doesn't exist yet, return None.
        """
        cur_dict = self.__retrieve_dict(iter_type, dict_type='best')

        if metric_key not in cur_dict:
            cur_dict[metric_key] = None

        return cur_dict[metric_key]

    def get_metric(self, iter_num, iter_type, metric_key):
        """
        Getter method for retrieving a mean aggregated metric for a certain iteration. 
        """
        cur_dict = self.__retrieve_dict(iter_type, dict_type='aggregate')

        metric_key = "mean_" + metric_key
        iterkey = 'iter_' + str(iter_num)
        if iterkey not in cur_dict or metric_key not in cur_dict[iterkey]:
            out_metric = None
        else:
            out_metric = cur_dict[iterkey][metric_key]

        return out_metric
    
    def get_all_metrics(self, iter_num, iter_type):
        """
        Getter method for retrieiving all the aggregated metrics for a certain iteration.
        """
        cur_dict = self.__retrieve_dict(iter_type, dict_type='aggregate')

        iterkey = 'iter_' + str(iter_num)
        if iterkey not in cur_dict:
            out_dict = None
        else:
            out_dict = cur_dict[iterkey]
        
        return out_dict

        
    def calc_iter_metrics(self, x_hat, x, iter_num, iter_type='train'):
        """
        Function for calculating and adding metrics from one iteration to the master.

        Args:
            x_hat: the proposed image(s). torch tensor with shape [N, C, H, W]
            x: ground truth image(s). torch tensor with shape [N, C, H, W]
            iter_num: the global iteration number. int
            iter_type: 'train', 'test', or 'val' - the type of metrics we are calculating
        """
        cur_dict = self.__retrieve_dict(iter_type) #validate and retrieve the right dict

        iter_metrics = get_all_metrics(x_hat, x, range = self.range, hparams=self.hparams) #calc the metrics

        self.__init_iter_dict(cur_dict, iter_num) #check that the iter dict is initialized
        
        self.__append_to_iter_dict(cur_dict, iter_metrics, iter_num) #add the values to the iter dict

        return
    
    def add_external_metrics(self, external_metrics, iter_num, iter_type='train'):
        """
        Function for adding a given dict of metrics to the given iteration.
        """
        cur_dict = self.__retrieve_dict(iter_type) #validate and retrieve the right dict

        self.__init_iter_dict(cur_dict, iter_num) #check that the iter dict is initialized
        
        self.__append_to_iter_dict(cur_dict, external_metrics, iter_num) #add the values to the iter dict

        return
    
    def aggregate_iter_metrics(self, iter_num, iter_type='train', return_best=False):
        """
        Called at the end of an iteration/epoch to find summary stats for all the metrics.
        If desired, returns a dict with the name and value of each metric from the iteration that 
            achieved their best value. If no metric had its best value, return None.
        """
        agg_dict = self.__retrieve_dict(iter_type, dict_type='aggregate') #validate and retrieve the right dicts
        raw_dict = self.__retrieve_dict(iter_type, dict_type='raw')
        best_dict = self.__retrieve_dict(iter_type, dict_type='best')

        self.__init_iter_dict(agg_dict, iter_num) #check that the iter dict is initialized
        self.__init_iter_dict(raw_dict, iter_num, should_exist=True) #make sure the corresponding dict exists in the raw

        #go through all the metrics in the raw data and aggregate them 
        iterkey = 'iter_' + str(iter_num)
        for key, value in raw_dict[iterkey].items():
            mean_key = "mean_" + key
            std_key = "std_" + key
            mean_value = np.mean(value)
            std_value = np.std(value)
            agg_dict[iterkey][mean_key] = mean_value
            agg_dict[iterkey][std_key] = std_value 
        
        if return_best:
            out_dict = None
        
        #aggregation is done, now we check if the aggregates values contain any bests
        for key, value in agg_dict[iterkey].items():
            if 'mean' not in key: #we are only interested in the mean values
                continue
            
            metric_key = key[5:] #strip "mean_" from the front of the key

            best = self.get_best(iter_type, metric_key)

            if best is None:
                best_dict[metric_key] = (iter_num, value)
            else:
                _, best_value = best
                best_flag = False

                if metric_key in ['psnr', 'ssim', 'ms-ssim']:
                    if best_value < value:
                        best_dict[metric_key] = (iter_num, value)
                        best_flag = True
                else:
                    if best_value > value:
                        best_dict[metric_key] = (iter_num, value)
                        best_flag = True

                if return_best and best_flag:
                    if out_dict is None:
                        out_dict = {}
                    out_dict[metric_key] = value
        
        if return_best:
            return out_dict
        else:
            return
    
    def add_metrics_to_tb(self, tb_logger, step, iter_type='train'):
        """
        Run through metrics and log everything there.
        For each type of metric, we want to log the train, val, and test metrics on the same plot.
        Intended to be called at the end of each type of iteration (train, test, val)
        """
        assert iter_type in ['train', 'val', 'test']

        raw_dict = self.__retrieve_dict(iter_type, dict_type='raw')
        agg_dict = self.__retrieve_dict(iter_type, dict_type='aggregate')
        best_dict = self.__retrieve_dict(iter_type, dict_type='best')

        iterkey ='iter_' + str(step)

        if iterkey not in raw_dict:
            print("\ncurrent iteration has not yet been logged\n")
            return
        
        for metric_type, metric_value in raw_dict[iterkey].items():
            for i, val in enumerate(metric_value):
                tb_logger.add_scalars("raw " + metric_type, {iter_type: val}, i)
        
        for metric_type, metric_value in agg_dict[iterkey].items():
            tb_logger.add_scalars(metric_type, {iter_type: metric_value}, step)
        
        for metric_type, metric_value in best_dict.items():
            tb_logger.add_scalars("best " + metric_type + " iter", {iter_type: metric_value[0]}, step)
            tb_logger.add_scalars("best " + metric_type + " value", {iter_type: metric_value[1]}, step)
        
        return
