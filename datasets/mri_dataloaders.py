from scipy.ndimage.interpolation import rotate,zoom
from torch.utils.data import Dataset, DataLoader
import glob
import os
import numpy as np
import torch
from tqdm import tqdm
import h5py
import sigpy as sp
import pickle as pkl
import sys

def get_mvue(kspace, s_maps):
    ''' Get mvue estimate from coil measurements '''
    return np.sum(sp.ifft(kspace, axes=(-1, -2)) * np.conj(s_maps), axis=-3) / np.sqrt(np.sum(np.square(np.abs(s_maps)), axis=-3))

class BrainMultiCoil(Dataset):
    def __init__(self, file_list, maps_dir, input_dir,
                 R=1,
                 image_size=384,
                 acs_size=26,
                 pattern='equispaced',
                 orientation='vertical'):
        # Attributes
        self.file_list    = file_list
        self.maps_dir     = maps_dir
        self.input_dir      = input_dir
        self.image_size = image_size
        self.R            = R
        self.pattern      = pattern
        self.orientation  = orientation

        # Access meta-data of each scan to get number of slices
        self.num_slices = np.zeros((len(self.file_list,)), dtype=int)
        for idx, file in enumerate(self.file_list):
            input_file = os.path.join(self.input_dir, os.path.basename(file))
            with h5py.File(input_file, 'r') as data:
                self.num_slices[idx] = int(np.array(data['kspace']).shape[0])

        # Create cumulative index for mapping
        self.slice_mapper = np.cumsum(self.num_slices) - 1 # Counts from '0'

    def __len__(self):
        return int(np.sum(self.num_slices)) # Total number of slices from all scans

    # Phase encode random mask generator
    def _get_mask(self, acs_lines=30, total_lines=384, R=1, pattern='random'):
        # Overall sampling budget
        num_sampled_lines = np.floor(total_lines / R)

        # Get locations of ACS lines
        # !!! Assumes k-space is even sized and centered, true for fastMRI
        center_line_idx = np.arange((total_lines - acs_lines) // 2,
                             (total_lines + acs_lines) // 2)

        # Find remaining candidates
        outer_line_idx = np.setdiff1d(np.arange(total_lines), center_line_idx)
        if pattern == 'random':
            # Sample remaining lines from outside the ACS at random
            random_line_idx = np.random.choice(outer_line_idx,
                       size=int(num_sampled_lines - acs_lines), replace=False)
        elif pattern == 'equispaced':
            # Sample equispaced lines
            # !!! Only supports integer for now
            random_line_idx = outer_line_idx[::int(R)]
        else:
            raise NotImplementedError('Mask pattern not implemented')

        # Create a mask and place ones at the right locations
        mask = np.zeros((total_lines))
        mask[center_line_idx] = 1.
        mask[random_line_idx] = 1.

        return mask

    # Cropping utility - works with numpy / tensors
    def _crop(self, x, wout, hout):
        w, h = x.shape[-2:]
        x1 = int(np.ceil((w - wout) / 2.))
        y1 = int(np.ceil((h - hout) / 2.))

        return x[..., x1:x1+wout, y1:y1+hout]

    def __getitem__(self, idx):
        print(idx)
        # Convert to numerical
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get scan and slice index
        # First scan for which index is in the valid cumulative range
        scan_idx = int(np.where((self.slice_mapper - idx) >= 0)[0][0])
        # Offset from cumulative range
        slice_idx = int(idx) if scan_idx == 0 else \
            int(idx - self.slice_mapper[scan_idx] + self.num_slices[scan_idx] - 1)

        # Load maps for specific scan and slice
        maps_file = os.path.join(self.maps_dir,
                                 os.path.basename(self.file_list[scan_idx]))
        with h5py.File(maps_file, 'r') as data:
            # Get maps
            s_maps = np.asarray(data['s_maps'][slice_idx])

        # Load raw data for specific scan and slice
        raw_file = os.path.join(self.input_dir,
                                os.path.basename(self.file_list[scan_idx]))
        with h5py.File(raw_file, 'r') as data:
            # Get maps
            gt_ksp = np.asarray(data['kspace'][slice_idx])
        # Crop extra lines and reduce FoV in phase-encode
        gt_ksp = sp.resize(gt_ksp, (gt_ksp.shape[0], gt_ksp.shape[1], self.image_size))

        # Reduce FoV by half in the readout direction
        gt_ksp = sp.ifft(gt_ksp, axes=(-2,))
        gt_ksp = sp.resize(gt_ksp, (gt_ksp.shape[0], self.image_size,
                                    gt_ksp.shape[2]))
        gt_ksp = sp.fft(gt_ksp, axes=(-2,)) # Back to k-space

        # Crop extra lines and reduce FoV in phase-encode
        s_maps = sp.fft(s_maps, axes=(-2, -1)) # These are now maps in k-space
        s_maps = sp.resize(s_maps, (
            s_maps.shape[0], s_maps.shape[1], self.image_size))

        # Reduce FoV by half in the readout direction
        s_maps = sp.ifft(s_maps, axes=(-2,))
        s_maps = sp.resize(s_maps, (s_maps.shape[0], self.image_size,
                                    s_maps.shape[2]))
        s_maps = sp.fft(s_maps, axes=(-2,)) # Back to k-space
        s_maps = sp.ifft(s_maps, axes=(-2, -1)) # Finally convert back to image domain

        # find mvue image
        gt_mvue = get_mvue(gt_ksp, s_maps)

        # Compute ACS size based on R factor and sample size
        total_lines = gt_ksp.shape[-1]
        if 1 < self.R <= 6:
            # Keep 8% of center samples
            acs_lines = np.floor(0.08 * total_lines).astype(int)
        else:
            # Keep 4% of center samples
            acs_lines = np.floor(0.04 * total_lines).astype(int)

        # Get a mask
        # mask = self._get_mask(acs_lines, total_lines,
        #                       self.R, self.pattern)
        # Mask k-space
        # if self.orientation == 'vertical':
        #     ksp = gt_ksp * mask[None, None, :]
        # elif self.orientation == 'horizontal':
        #     ksp = gt_ksp * mask[None, :, None]
        # else:
        #     raise NotImplementedError

        ksp = gt_ksp
        # find mvue image
        aliased_mvue = get_mvue(ksp, s_maps)

        scale_factor = np.percentile(np.abs(aliased_mvue), 99)
        ksp /= scale_factor
        aliased_mvue /= scale_factor

        gt_mvue_scale_factor = np.percentile(np.abs(gt_mvue),99)
        gt_mvue /= gt_mvue_scale_factor

        s_maps_scale = np.sqrt(np.sum(np.square(np.abs(s_maps)), axis=-3))
        # print(s_maps_scale)
        ksp /= s_maps_scale
        aliased_mvue /= s_maps_scale
        gt_mvue /= s_maps_scale
        s_maps /= s_maps_scale

        # Apply ACS-based instance scaling
        aliased_mvue_two_channel = np.float16(np.zeros((2,) + aliased_mvue.shape))
        aliased_mvue_two_channel[0] = np.float16(np.real(aliased_mvue))
        aliased_mvue_two_channel[1] = np.float16(np.imag(aliased_mvue))

        gt_mvue_two_channel = np.float16(np.zeros((2,) + gt_mvue.shape))
        gt_mvue_two_channel[0] = np.float16(np.real(gt_mvue))
        gt_mvue_two_channel[1] = np.float16(np.imag(gt_mvue))


        # Output
        sample = {
                  'ksp': ksp,
                  's_maps': s_maps,
                  # 'mask': mask,
                  'aliased_image': aliased_mvue_two_channel.astype(np.float32),
                  'gt_image': gt_mvue_two_channel.astype(np.float32),
                  'scale_factor': scale_factor.astype(np.float32),
                  # Just for feedback
                  'scan_idx': scan_idx,
                  'slice_idx': slice_idx}
        print(ksp.shape)
        return sample, idx

class KneesMultiCoil(BrainMultiCoil):
    def __init__(self, file_list, maps_dir, input_dir,
                 R=1,
                 image_size=320,
                 acs_size=26,
                 pattern='random',
                 orientation='vertical'):
        super(KneesMultiCoil, self).__init__(file_list, maps_dir, input_dir, R,
                                             image_size, acs_size, pattern,
                                             orientation)

class KneesSingleCoil(Dataset):
    def __init__(self,
                 files,
                 image_size,
                 R,
                 pattern,
                 orientation
    ):
        self.file_list = sorted(files)
        self.image_size = image_size
        self.R = R
        self.pattern = pattern
        self.orientation = orientation


        # name = os.path.splitext(os.path.basename(self._path))[0]
        # raw_shape = [len(self._file_list)] + list(self.__getitem__(0)[0].shape)
        # if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
        #     raise IOError('Image files do not match the specified resolution')

    # @property
    # def num_slices(self):
    #     num_slices = np.zeros((len(self._file_list,)), dtype=int)
    #     for idx, file in enumerate(self._file_list):
    #         with h5py.File(os.path.join(file), 'r') as data:
    #             num_slices[idx] = int(data['kspace'].shape[0])
    #     return num_slices
    #
    # @property
    # def slice_mapper(self):
    #     return np.cumsum(self.num_slices) - 1 # Counts from '0'
    #
    # def __len__(self):
    #     return int(np.sum(self.num_slices)) # Total number of slices from all scans

    # Phase encode random mask generator
    def _get_mask(self, acs_lines=30, total_lines=384, R=1, pattern='random'):
        # Overall sampling budget
        num_sampled_lines = np.floor(total_lines / R)

        # Get locations of ACS lines
        # !!! Assumes k-space is even sized and centered, true for fastMRI
        center_line_idx = np.arange((total_lines - acs_lines) // 2,
                             (total_lines + acs_lines) // 2)

        # Find remaining candidates
        outer_line_idx = np.setdiff1d(np.arange(total_lines), center_line_idx)
        if pattern == 'random':
            # Sample remaining lines from outside the ACS at random
            random_line_idx = np.random.choice(outer_line_idx,
                       size=int(num_sampled_lines - acs_lines), replace=False)
        elif pattern == 'equispaced':
            # Sample equispaced lines
            # !!! Only supports integer for now
            random_line_idx = outer_line_idx[::int(R)]
        else:
            raise NotImplementedError('Mask pattern not implemented')

        # Create a mask and place ones at the right locations
        mask = np.zeros((total_lines))
        mask[center_line_idx] = 1.
        mask[random_line_idx] = 1.

        return mask

    def __len__(self):
        return int(len(self.file_list) * 5) # Total number of slices from all scans

    # Cropping utility - works with numpy / tensors
    def _crop(self, x, wout, hout):
        w, h = x.shape[-2:]
        x1 = int(np.ceil((w - wout) / 2.))
        y1 = int(np.ceil((h - hout) / 2.))

        return x[..., x1:x1+wout, y1:y1+hout]

    def __getitem__(self, idx):
        # Convert to numerical
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get scan and slice index
        # we only use the 5 central slices
        scan_idx = idx // 5 # int(np.where((self.slice_mapper - idx) >= 0)[0][0])
        # Offset from cumulative range
        slice_idx = idx % 5
        # slice_idx = int(idx) if scan_idx == 0 else \
        #     int(idx - self.slice_mapper[scan_idx] + self.num_slices[scan_idx] - 1)

        # Load specific slice from specific scan
        with h5py.File(os.path.join(self.file_list[scan_idx]), 'r') as data:
            # Get RSS and scaling factor
            num_slices = int(data['kspace'].shape[0])
            slice_idx_shifted = (num_slices // 2) - 2 + slice_idx
            gt_ksp      = np.asarray(data['kspace'][slice_idx_shifted])
        # print(f'name: {self.file_list[scan_idx]}, slice:{slice_idx_shifted}')

        # kspace = self._crop(kspace, self._resolution, self._resolution)
        gt_image = sp.ifft(gt_ksp, axes=(-2,-1))
        gt_image = sp.resize(gt_image, (self.image_size, gt_image.shape[-1]))

        gt_ksp = sp.fft(gt_image, axes=(-2,-1))
        gt_ksp = sp.resize(gt_ksp, (self.image_size, self.image_size))

        gt_image = sp.ifft(gt_ksp, axes=(-2,-1))

        # Compute ACS size based on R factor and sample size
        total_lines = gt_ksp.shape[-1]
        if 1 < self.R <= 6:
            # Keep 8% of center samples
            acs_lines = np.floor(0.08 * total_lines).astype(int)
        else:
            # Keep 4% of center samples
            acs_lines = np.floor(0.04 * total_lines).astype(int)

        # Get a mask
        mask = self._get_mask(acs_lines, total_lines,
                              self.R, self.pattern)
        # Mask k-space
        # if self.orientation == 'vertical':
        #     ksp = gt_ksp * mask[None, :]
        # elif self.orientation == 'horizontal':
        #     ksp = gt_ksp * mask[:, None]
        # else:
        #     raise NotImplementedError
        ksp = gt_ksp

        aliased_image = sp.ifft(ksp, axes=(-2,-1))
        scale_factor = np.percentile(np.abs(aliased_image), 99)
        ksp /= scale_factor
        aliased_image /= scale_factor

        gt_image_scale_factor = np.percentile(np.abs(gt_image),99)
        gt_image /= gt_image_scale_factor

        # Apply ACS-based instance scaling
        aliased_image_two_channel = np.float16(np.zeros((2,) + aliased_image.shape))
        aliased_image_two_channel[0] = np.float16(np.real(aliased_image))
        aliased_image_two_channel[1] = np.float16(np.imag(aliased_image))

        gt_image_two_channel = np.float16(np.zeros((2,) + gt_image.shape))
        gt_image_two_channel[0] = np.float16(np.real(gt_image))
        gt_image_two_channel[1] = np.float16(np.imag(gt_image))
        sample = {
                  'ksp': ksp,
                  'mask': mask,
                  'aliased_image': aliased_image_two_channel.astype(np.float32),
                  'gt_image': gt_image_two_channel.astype(np.float32),
                  'scale_factor': scale_factor.astype(np.float32),
                  # Just for feedback
                  'scan_idx': scan_idx,
                  'slice_idx': slice_idx_shifted}
        return sample, idx


