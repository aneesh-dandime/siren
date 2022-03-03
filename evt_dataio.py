import numpy as np
import skvideo.io
import torch
from torch.utils.data import Dataset, DataLoader


def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.'''
    if isinstance(sidelen, int):
        sidelen = dim * (sidelen,)

    if dim == 2:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[0, :, :, 0] = pixel_coords[0, :, :, 0] / (sidelen[0] - 1)
        pixel_coords[0, :, :, 1] = pixel_coords[0, :, :, 1] / (sidelen[1] - 1)
    elif dim == 3:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1], :sidelen[2]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[..., 0] = pixel_coords[..., 0] / max(sidelen[0] - 1, 1)
        pixel_coords[..., 1] = pixel_coords[..., 1] / (sidelen[1] - 1)
        pixel_coords[..., 2] = pixel_coords[..., 2] / (sidelen[2] - 1)
    else:
        raise NotImplementedError('Not implemented for dim=%d' % dim)

    pixel_coords -= 0.5
    pixel_coords *= 2.
    pixel_coords = pixel_coords[0, :, :, :, :]
    return pixel_coords


class Video(Dataset):
    def __init__(self, path_to_video):
        super().__init__()
        if 'npy' in path_to_video:
            self.vid = np.load(path_to_video)
        elif 'mp4' in path_to_video:
            self.vid = skvideo.io.vread(path_to_video, as_grey=True).astype(np.single) / 255.

        self.shape = self.vid.shape[:-1]
        self.channels = self.vid.shape[-1]

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.vid


class Implicit3DWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset, sidelength=None, sample_fraction=1.):

        if isinstance(sidelength, int):
            sidelength = 3 * (sidelength,)

        self.dataset = dataset
        self.mgrid = get_mgrid(sidelength, dim=3)
        self.data = (self.dataset[0] - 0.5) / 0.5
        self.sample_fraction = sample_fraction
        self.total_points = self.mgrid.shape[0] * self.mgrid.shape[1] * self.mgrid.shape[2]
        self.N_samples = int(self.sample_fraction * self.total_points)
        print(self.N_samples)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if self.sample_fraction < 1.:
            fs = np.random.randint(0, self.mgrid.shape[0] - 1, (self.N_samples,))
            xs = np.random.randint(0, self.mgrid.shape[1], (self.N_samples,))
            ys = np.random.randint(0, self.mgrid.shape[2], (self.N_samples,))
            coords_first = self.mgrid[(fs, xs, ys)]
            data_first = self.data[(fs, xs, ys)]
            coords_next_frame = self.mgrid[(fs + 1, xs, ys)]
            data_next_frame = self.data[(fs + 1, xs, ys)]
            coords = np.concatenate((coords_first, coords_next_frame))
            data = np.concatenate((data_first, data_next_frame))
        else:
            coords = self.mgrid.view(-1, 3)
            data = self.data.view(-1, self.dataset.channels)

        in_dict = {'idx': idx, 'coords': coords}
        gt_dict = {'img': data}

        return in_dict, gt_dict


class Implicit3DWrapper2(torch.utils.data.Dataset):
    def __init__(self, dataset, sidelength=None, sample_fraction=1.):

        if isinstance(sidelength, int):
            sidelength = 3 * (sidelength,)

        self.dataset = dataset
        self.mgrid = get_mgrid(sidelength, dim=3)
        self.data = (self.dataset[0] - 0.5) / 0.5
        self.sample_fraction = sample_fraction
        self.total_points = self.mgrid.shape[0] * self.mgrid.shape[1] * self.mgrid.shape[2]
        self.N_samples = int(self.sample_fraction * self.total_points)
        print(self.N_samples)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if self.sample_fraction < 1.:
            fs = np.random.randint(1, self.mgrid.shape[0] - 1, (self.N_samples,))
            xs = np.random.randint(0, self.mgrid.shape[1], (self.N_samples,))
            ys = np.random.randint(0, self.mgrid.shape[2], (self.N_samples,))
            coords_first = self.mgrid[(fs - 1, xs, ys)]
            data_first = self.data[(fs - 1, xs, ys)]
            coords_next_frame = self.mgrid[(fs + 1, xs, ys)]
            data_next_frame = self.data[(fs + 1, xs, ys)]
            coords = np.concatenate((coords_first, coords_next_frame))
            data = np.concatenate((data_first, data_next_frame))
        else:
            coords = self.mgrid.view(-1, 3)
            data = self.data.view(-1, self.dataset.channels)

        in_dict = {'idx': idx, 'coords': coords}
        gt_dict = {'img': data}

        return in_dict, gt_dict


class Implicit3DWrapperLinear(torch.utils.data.Dataset):
    def __init__(self, dataset, sidelength=None, sample_fraction=0.5):

        if isinstance(sidelength, int):
            sidelength = 3 * (sidelength,)

        self.dataset = dataset
        self.mgrid = get_mgrid(sidelength, dim=3)
        self.data = (self.dataset[0] - 0.5) / 0.5
        self.len = self.dataset.shape[0] - 2
        self.sample_fraction = sample_fraction
        self.sample_x = int(self.sample_fraction * self.dataset.shape[1])
        self.sample_y = int(self.sample_fraction * self.dataset.shape[2])

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        start_x = np.random.randint(0, self.mgrid.shape[1] - self.sample_x)
        end_x = start_x + self.sample_x
        start_y = np.random.randint(0, self.mgrid.shape[2] - self.sample_y)
        end_y = start_y + self.sample_y
        coords_first = self.mgrid[idx + 1 - 1, start_x: end_x, start_y: end_y, :].reshape(-1, 3)
        coords_last = self.mgrid[idx + 1 + 1, start_x: end_x, start_y: end_y, :].reshape(-1, 3)
        data_first = self.data[idx + 1 - 1, start_x: end_x, start_y: end_y, :].reshape(-1, self.dataset.channels)
        data_last = self.data[idx + 1 + 1, start_x: end_x, start_y: end_y, :].reshape(-1, self.dataset.channels)
        coords = np.concatenate((coords_first, coords_last))
        data = np.concatenate((data_first, data_last))

        in_dict = {'idx': idx, 'coords': coords}
        gt_dict = {'img': data}

        return in_dict, gt_dict


if __name__ == '__main__':
    video_path = './data/cat_video.mp4'
    vid_dataset = Video(video_path)
    coord_dataset = Implicit3DWrapperLinear(vid_dataset, vid_dataset.shape)
    dataloader = DataLoader(coord_dataset, shuffle=True, batch_size=1, pin_memory=True, num_workers=0)
    dataiter = iter(dataloader)
    inp, gt = dataiter.next()
    print(inp['coords'].shape)
    print(gt['img'].shape)

