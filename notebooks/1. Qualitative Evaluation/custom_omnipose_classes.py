from cellpose_omni import models
from cellpose_omni.models import deprecation_warning_cellprob_dist_threshold
import os, sys, time, shutil, tempfile, datetime, pathlib, subprocess
from pathlib import Path
import numpy as np
# from tqdm import trange, tqdm
from tqdm.auto import tqdm, trange
import omnipose
from omnipose.logger import setup_logger
omnipose_logger = setup_logger('core')
from urllib.parse import urlparse
import torch
from torch import nn, distributed, multiprocessing, optim

# from torch.nn.parallel import DistributedDataParallel as DDP

from scipy.ndimage import gaussian_filter, zoom

import logging
models_logger = logging.getLogger(__name__)

from cellpose_omni import transforms, dynamics, utils, plot
from cellpose_omni.core import UnetModel, assign_device, check_mkl, MXNET_ENABLED, parse_model_string
from cellpose_omni.io import OMNI_INSTALLED
from omnipose.gpu import empty_cache, ARM #, custom_nonzero_cuda
from omnipose.utils import hysteresis_threshold
import ncolor

from torchvf.numerics import interp_vf, ivp_solver

class CellposeModelCustomEuler(models.CellposeModel):
    def __init__(self, gpu=False, pretrained_model=False,
                 model_type=None, net_avg=True, use_torch=True,
                 diam_mean=30., device=None,
                 residual_on=True, style_on=True, concatenation=False,
                 nchan=1, nclasses=None, dim=2, omni=True, logits=False,
                 nsample=4, # number of up/downsampling layers
                 checkpoint=False, dropout=False,
                 kernel_size=2,scale_factor=2,dilation=1,
                 allow_blank_masks=False):
        super().__init__(
            gpu=gpu,
            pretrained_model=pretrained_model,
            model_type=model_type,
            net_avg=net_avg,
            use_torch=use_torch,
            diam_mean=diam_mean,
            device=device,
            residual_on=residual_on,
            style_on=style_on,
            concatenation=concatenation,
            nchan=nchan,
            nclasses=nclasses,
            dim=dim,
            omni=omni,
            logits=logits,
            nsample=nsample,
            checkpoint=checkpoint,
            dropout=dropout,
            kernel_size=kernel_size,
            scale_factor=scale_factor,
            dilation=dilation,
            allow_blank_masks=allow_blank_masks)
        print('CellposeModelCustomEuler initialised')

    def eval(self, x, batch_size=8, indices=None, channels=None, channel_axis=None,
             z_axis=None, normalize=True, invert=False,
             rescale=None, diameter=None, do_3D=False, anisotropy=None, net_avg=True,
             augment=False, tile=False, tile_overlap=0.1, bsize=224, num_workers=8,
             loader_batch_size=1,  # for torch dataloader
             resample=True, interp=True, cluster=False, hdbscan=False, suppress=None,
             boundary_seg=False, affinity_seg=False, despur=False,
             flow_threshold=0.4, mask_threshold=0.0, diam_threshold=12., niter=None,
             cellprob_threshold=None, dist_threshold=None, flow_factor=5.0,
             compute_masks=True, min_size=15, max_size=None, stitch_threshold=0.0,
             progress=None, show_progress=True,
             omni=False, calc_trace=False, verbose=False, transparency=False,
             loop_run=False, model_loaded=False, hysteresis=True, supr_fac=1):
        """
            Evaluation for CellposeModel. Segment list of images x, or 4D array - Z x nchan x Y x X

            Parameters
            ----------
            x: list or array of images
                can be list of 2D/3D/4D images, or array of 2D/3D/4D images

            batch_size: int (optional, default 8)
                number of 224x224 patches to run simultaneously on the GPU
                (can make smaller or bigger depending on GPU memory usage)

            channels: list (optional, default None)
                list of channels, either of length 2 or of length number of images by 2.
                First element of list is the channel to segment (0=grayscale, 1=red, 2=green, 3=blue).
                Second element of list is the optional nuclear channel (0=none, 1=red, 2=green, 3=blue).
                For instance, to segment grayscale images, input [0,0]. To segment images with cells
                in green and nuclei in blue, input [2,3]. To segment one grayscale image and one
                image with cells in green and nuclei in blue, input [[0,0], [2,3]].

            channel_axis: int (optional, default None)
                if None, channels dimension is attempted to be automatically determined

            z_axis: int (optional, default None)
                if None, z dimension is attempted to be automatically determined

            normalize: bool (default, True)
                normalize data so 0.0=1st percentile and 1.0=99th percentile of image intensities in each channel

            invert: bool (optional, default False)
                invert image pixel intensity before running network

            rescale: float (optional, default None)
                resize factor for each image, if None, set to 1.0

            diameter: float (optional, default None)
                diameter for each image (only used if rescale is None),
                if diameter is None, set to diam_mean

            do_3D: bool (optional, default False)
                set to True to run 3D segmentation on 4D image input

            anisotropy: float (optional, default None)
                for 3D segmentation, optional rescaling factor (e.g. set to 2.0 if Z is sampled half as dense as X or Y)

            net_avg: bool (optional, default True)
                runs the 4 built-in networks and averages them if True, runs one network if False

            augment: bool (optional, default False)
                tiles image with overlapping tiles and flips overlapped regions to augment

            tile: bool (optional, default True)
                tiles image to ensure GPU/CPU memory usage limited (recommended)

            tile_overlap: float (optional, default 0.1)
                fraction of overlap of tiles when computing flows

            resample: bool (optional, default True)
                run dynamics at original image size (will be slower but create more accurate boundaries)

            interp: bool (optional, default True)
                interpolate during 2D dynamics (not available in 3D)
                (in previous versions it was False)

            flow_threshold: float (optional, default 0.4)
                flow error threshold (all cells with errors below threshold are kept) (not used for 3D)

            mask_threshold: float (optional, default 0.0)
                all pixels with value above threshold kept for masks, decrease to find more and larger masks

            dist_threshold: float (optional, default None) DEPRECATED
                use mask_threshold instead

            cellprob_threshold: float (optional, default None) DEPRECATED
                use mask_threshold instead

            compute_masks: bool (optional, default True)
                Whether or not to compute dynamics and return masks.
                This is set to False when retrieving the styles for the size model.

            min_size: int (optional, default 15)
                minimum number of pixels per mask, can turn off with -1

            stitch_threshold: float (optional, default 0.0)
                if stitch_threshold>0.0 and not do_3D, masks are stitched in 3D to return volume segmentation

            progress: pyqt progress bar (optional, default None)
                to return progress bar status to GUI

            omni: bool (optional, default False)
                use omnipose mask reconstruction features

            calc_trace: bool (optional, default False)
                calculate pixel traces and return as part of the flow

            verbose: bool (optional, default False)
                turn on additional output to logs for debugging

            transparency: bool (optional, default False)
                modulate flow opacity by magnitude instead of brightness (can use flows on any color background)

            loop_run: bool (optional, default False)
                internal variable for determining if model has been loaded, stops model loading in loop over images

            model_loaded: bool (optional, default False)
                internal variable for determining if model has been loaded, used in __main__.py

            Returns
            -------
            masks: list of 2D arrays, or single 3D array (if do_3D=True)
                labelled image, where 0=no masks; 1,2,...=mask labels

            flows: list of lists 2D arrays, or list of 3D arrays (if do_3D=True)
                flows[k][0] = 8-bit RGb phase plot of flow field
                flows[k][1] = flows at each pixel
                flows[k][2] = scalar cell probability (Cellpose) or distance transform (Omnipose)
                flows[k][3] = boundary output (nonempty for Omnipose)
                flows[k][4] = final pixel locations after Euler integration
                flows[k][5] = pixel traces (nonempty for calc_trace=True)

            styles: list of 1D arrays of length 64, or single 1D array (if do_3D=True)
                style vector summarizing each image, also used to estimate size of objects in image

        """

        if cellprob_threshold is not None or dist_threshold is not None:
            mask_threshold = deprecation_warning_cellprob_dist_threshold(cellprob_threshold, dist_threshold)

        # images are given has a list, especially when heterogeneous in shape
        is_grey = np.sum(channels) == 0
        slice_ndim = self.dim + do_3D + (self.nchan > 1 and not is_grey) + (channel_axis is not None)
        # the logic here needs to be updated to account for the fact that images may not already match the expected dims
        # and channels, namely mono channel might have a 2-channel model. I should just check for if the number of channels could
        # possibly match, and warn that internal conversion will happen or may break...
        is_list = isinstance(x, list)
        is_stack = is_image = False

        if verbose:
            models_logger.info(
                f'is_grey {is_grey}, slice_ndim {slice_ndim}, dim {self.dim}, nchan {self.nchan}, is_list {is_list}')

        if isinstance(x, np.ndarray):
            # [0,0] is a special instance where we want to run the model on a single channel
            dim_diff = x.ndim - slice_ndim
            opt = np.array([0, 1])  # -is_grey
            is_image, is_stack = [dim_diff == i for i in opt]
            correct_shape = dim_diff in opt

        if verbose:
            models_logger.info(f'is_image {is_image}, is_stack {is_stack}, is_list {is_list}')
        # print('a1',interp,hysteresis,calc_trace)

        # allow for a dataset to be passed so that we can do batches
        # will be defined in omnipose.data.train_set
        is_dataset = isinstance(x, torch.utils.data.Dataset)  # if using eval_set
        if is_dataset:
            correct_shape = True  # assume the dataset has the right shape

        if not (is_list or is_stack or is_dataset or is_image or loop_run):
            models_logger.warning('input images must be a list of images, array of images, or dataloader')
        else:
            if is_list:
                correct_shape = np.all([x[i].squeeze().ndim == slice_ndim for i in range(len(x))])

            if not correct_shape:
                # print(slice_ndim,x.ndim,is_list,is_stack)
                models_logger.warning(
                    'input images do not match the expected number of dimensions ({}) \nand channels ({}) of model.'.format(
                        self.dim, self.nchan))

        if verbose and (is_dataset or not (is_list or is_stack)):
            models_logger.info(
                'Evaluating with flow_threshold %0.2f, mask_threshold %0.2f' % (flow_threshold, mask_threshold))
            if omni:
                models_logger.info(f'using omni model, cluster {cluster}')

        # Note: dataset is finetuned for basic omnipose usage. No styles are returned, some options may not be supported.
        if is_dataset:

            if verbose:
                models_logger.warning('Using dataset evaluation branch. Some options not yet supported.')

            # set the tile parameter in dataset
            x.tile = tile

            # set the rescale parameter in dataset
            x.rescale = 1.0 if rescale is None else rescale

            # sample indices to evaluate
            indices = list(range(len(x))) if indices is None else indices

            # the sequential batch sampler gives us a set of indices in sequence, like 0-5, 6-11, etc.
            sampler = torch.utils.data.sampler.BatchSampler(omnipose.data.sampler(indices),
                                                            batch_size=loader_batch_size,
                                                            drop_last=False)

            params = {'batch_size': 1,  # this batch size is more like how many worker batches to aggregate
                      #   'shuffle': False, # use sampler instead
                      'collate_fn': x.collate_fn,
                      'pin_memory': False,  # only useful for CPU tensors
                      'num_workers': num_workers,
                      'sampler': sampler,  # iterabledataset does not need this
                      'persistent_workers': True if num_workers > 0 else False,
                      #   'multiprocessing_context': 'spawn' if num_workers>0 else None, # consider 'forkserver'
                      'multiprocessing_context': 'fork' if num_workers > 0 else None,

                      'prefetch_factor': batch_size if num_workers > 0 else None
                      }

            loader = torch.utils.data.DataLoader(x, **params)
            dist, dP, bd, masks, bounds, p, tr, affinity, flow_RGB = [], [], [], [], [], [], [], [], []

            # I think the loader can at least do all the preprocessing work it will take to figure out
            # padding and stitching and slicing
            progress_bar = tqdm(total=len(indices), disable=not show_progress)
            for batch, inds, subs in loader:
                batch = batch.to(self.device)  # move to GPU

                shape = batch.shape
                nimg = batch.shape[0]
                nchan = batch.shape[1]
                shape = batch.shape[-(self.dim + 1):]  # nclasses, Y, X
                resize = shape[-self.dim:] if not resample else None

                # define the slice needed to get rid of padding required for net downsamples
                slc = [slice(0, s + 1) for s in shape]
                slc[-(self.dim + 1)] = slice(0, self.nclasses + 1)
                for k in range(1, self.dim + 1):
                    slc[-k] = slice(subs[-k][0], subs[-k][-1] + 1)
                slc = tuple(slc)

                # catch cases where the images are 1-channel
                # but the model is 2 channel
                # if self.nchan-nchan:
                #     print('padding with extra chan dd',batch)
                #     batch = torch.cat([batch,torch.zeros_like(batch)],dim=1)#.permute(0,2,3,1)
                #     print('now',batch)

                # batch = torch.cat([batch,batch],dim=1)
                # batch = torch.cat([torch.zeros_like(batch),batch],dim=1)

                # run the network on the batch
                # yf, style = self.network(batch)

                with torch.no_grad():  # this should also be in self.network, redundant?
                    # self.net.eval() # was missing this - some layers behave differently without it
                    # actually, self.network should have it now

                    if tile:
                        yf = x._run_tiled(batch, self,
                                          batch_size=batch_size,
                                          bsize=bsize,
                                          augment=augment,
                                          tile_overlap=tile_overlap)  # .unsqueeze(0)
                    else:
                        yf = self.network(batch, to_numpy=False)[0]
                        # yf = self.net(batch)[0] go back to this if error

                    del batch
                    # print('need to add normalization / invert /rescale options in dataloader')

                # slice out padding
                yf = yf[(Ellipsis,) + slc]

                # rescale and resample
                if resample and rescale not in [None, 1.0, 0]:
                    yf = omnipose.data.torch_zoom(yf, 1 / rescale)

                # compared to the usual per-image pipeline, this one will not support cellpose or u-net
                flow_pred = yf[:, :self.dim]
                dist_pred = yf[:, self.dim]  # scalar field always after the vector field output
                # might need to invert the log trasnformatiion here

                if self.nclasses >= self.dim + 2:
                    bd_pred = yf[:, self.dim + 1]
                else:
                    bd_pred = torch.empty(nimg)

                # clear from memory
                del yf

                # I made a vastly faster implementation using pytorch
                rgb = omnipose.plot.rgb_flow(flow_pred, transparency=transparency)

                # I implemented hysteresis with just pytorch
                # it is faster than skimage with larger batches, but not by much
                # it does better in thin sections, however (though might be broken skeleton fragments)
                # I might just replace the main branch code with this
                if hysteresis:
                    foreground = hysteresis_threshold(dist_pred.unsqueeze(1), mask_threshold - 1,
                                                      mask_threshold).squeeze(dim=1)
                else:
                    foreground = dist_pred >= mask_threshold
                    # print('add flag')

                # print('fg_here',torch.sum(foreground))

                # vf = interp_vf(flow_pred/5., mode = "nearest_batched")
                # initial_points = init_values_semantic(foreground, device=self.device)

                shape = flow_pred.shape
                B = shape[0]
                dims = shape[-self.dim:]

                coords = [torch.arange(0, l, device=self.device) for l in dims]
                mesh = torch.meshgrid(coords, indexing="ij")
                init_shape = [B, 1] + ([1] * len(dims))
                initial_points = torch.stack(mesh, dim=0)  # torchvf flips with mesh[::-1]
                initial_points = initial_points.repeat(init_shape).float()

                # final_points = ivp_solver(vf,initial_points,
                #                         dx = 1,
                #                         n_steps = 8,
                #                         solver = "euler")[-1]

                # these three are equivalent
                coords = torch.nonzero(foreground, as_tuple=True)
                # coords = custom_nonzero_cuda(foreground.squeeze())
                # coords = torch.where(foreground.squeeze())

                # this block works

                # # Assuming foreground is a boolean tensor of shape (B, D1, D2, ..., DN)
                # fg = foreground.squeeze()  # Now fg has shape (B, D1, D2, ..., DN)

                # # Create a grid of indices
                # grids = torch.meshgrid([torch.arange(size, device=fg.device) for size in fg.shape])

                # # Stack the grids to create an index mesh
                # index_mesh = torch.stack(grids, dim=0)  # Now index_mesh has shape (N+1, B, D1, D2, ..., DN)

                # # Move index_mesh to the same device as foreground
                # index_mesh = index_mesh.to(fg.device)

                # # Use the boolean tensor to index into the index mesh
                # selected_indices = index_mesh[:, fg]
                # coords = tuple(selected_indices)

                # fg = foreground.squeeze()  # Now fg has shape (B, D1, D2, ..., DN)

                # # Create a grid of indices
                # grids = torch.meshgrid([torch.arange(size, device=fg.device) for size in fg.shape])

                # # Reshape each grid to have shape (-1)
                # reshaped_grids = [grid.reshape(-1) for grid in grids]

                # # Convert the reshaped grids to a tuple of indices
                # selected_indices = tuple(reshaped_grids)

                # # print(len(reshaped_grids),reshaped_grids[0].shape,reshaped_grids)

                # coords = tuple(selected_indices)

                # add to output lists
                dP.extend(self._from_device(flow_pred))
                dist.extend(self._from_device(dist_pred))
                bd.extend(self._from_device(bd_pred))
                flow_RGB.extend(self._from_device(rgb))

                if torch.any(foreground):
                    cell_px = (Ellipsis,) + coords[-self.dim:]
                    if niter is None:
                        # niter = omnipose.core.get_niter(dist_pred).cpu()
                        # int(diameters(foreground,dist_pred)/(1+affinity_seg))
                        niter = int(
                            2 * (self.dim + 1) * torch.mean(dist_pred[(Ellipsis,) + coords]) / (1 + affinity_seg))
                        if verbose:
                            models_logger.info('niter set to %d' % niter)

                    final_points = initial_points.clone()
                    final_p, traced_p = steps_batch(initial_points[cell_px],
                                                                  flow_pred / 5.,
                                                                  # <<<<<<<<<<< add support for other options here
                                                                  niter=niter,
                                                                  omni=omni,
                                                                  suppress=suppress,
                                                                  interp=interp,
                                                                  verbose=verbose,
                                                                  calc_trace=calc_trace,
                                                                  supr_fac=supr_fac)

                    final_points[cell_px] = final_p.squeeze()

                    if affinity_seg:
                        steps, inds, idx, fact, sign = omnipose.utils.kernel_setup(self.dim)
                        supporting_inds = omnipose.utils.get_supporting_inds(steps)
                        affinity_graph = omnipose.core._get_affinity_torch(initial_points,
                                                                           final_points,
                                                                           flow_pred / 5.,
                                                                           # <<<<<<<<<<< add support for other options here
                                                                           dist_pred,
                                                                           foreground,
                                                                           steps,
                                                                           fact,
                                                                           inds,
                                                                           supporting_inds,
                                                                           niter,
                                                                           )

                    # cast to CPU for compute_masks
                    final_points = self._from_device(final_points)
                    traced_p = self._from_device(traced_p) if traced_p is not None else [None] * B
                    affinity_graph = self._from_device(affinity_graph).swapaxes(0, 1) if affinity_seg else [None] * B
                    foreground = self._from_device(foreground)
                    dist_pred = self._from_device(dist_pred)
                    flow_pred = self._from_device(flow_pred)
                    bd_pred = self._from_device(bd_pred)
                    rgb = self._from_device(rgb)

                    # can loop through batch and run compute_masks
                    for iscell, disti, dPi, bdi, agi, pts, trp in zip(foreground, dist_pred, flow_pred, bd_pred,
                                                                      affinity_graph, final_points, traced_p):
                        parallel = 1
                        coords = np.nonzero(iscell)
                        # print('agi 33',agi.shape, affinity_graph.shape, np.sum(iscell), np.stack(coords).shape)
                        # agi = None
                        # print('torch computed affinity not quite ready yet ')
                        # print('PARALLEL', parallel)
                        # NOW THAT THE trajectories are "WORKING", I need to add the parallel affinity here
                        outputs = omnipose.core.compute_masks(dPi, disti,
                                                              affinity_graph=agi[
                                                                  (Ellipsis,) + coords] if agi is not None else agi,
                                                              bd=bdi,
                                                              p=pts.squeeze() if parallel else None,
                                                              coords=np.stack(coords),
                                                              iscell=iscell if parallel else None,
                                                              niter=niter,
                                                              rescale=rescale,
                                                              resize=resize,
                                                              min_size=min_size,
                                                              max_size=max_size,
                                                              mask_threshold=mask_threshold,
                                                              diam_threshold=diam_threshold,
                                                              flow_threshold=flow_threshold,
                                                              flow_factor=flow_factor,
                                                              interp=interp,
                                                              cluster=cluster,
                                                              hdbscan=hdbscan,
                                                              boundary_seg=boundary_seg,
                                                              affinity_seg=affinity_seg,
                                                              despur=despur,
                                                              calc_trace=calc_trace,
                                                              verbose=verbose,
                                                              use_gpu=self.gpu,
                                                              device=self.device,
                                                              nclasses=self.nclasses,
                                                              dim=self.dim)

                        masks.append(outputs[0])
                        p.append(outputs[1])
                        # tr.append(outputs[2])
                        tr.append(trp)
                        bounds.append(outputs[3])
                        affinity.append(outputs[4])

                        progress_bar.update()
                        empty_cache()



                else:
                    progress_bar.update()
                    empty_cache()
                    models_logger.info('no cell pixels found')
                    masks = np.zeros((B,) + dims)
                    bounds = np.zeros((B,) + dims)
                    affinity = [None for _ in range(B)]
                    tr = [None for _ in range(B)]
                    p = [None for _ in range(B)]

            masks = np.array(masks)
            bounds = np.array(bounds)
            p = np.array(p)
            tr = np.array(tr)
            ret = [masks, dP, dist, p, bd, tr, affinity, bounds, flow_RGB]

            progress_bar.close()

            for r in ret:
                r.squeeze() if isinstance(r, np.ndarray) else r

                # the flow list stores:
            # (1) RGB representation of flows
            # (2) flow components
            # (3) cellprob (cp) or distance field (op)
            # (4) pixel coordinates after Euler integration
            # (5) boundary output (nclasses=4)
            # (6) pixel trajectories during Euler integation (trace=True)
            # (7) nstep_by_npix affinity graph
            # (8) binary boundary map
            # 5-8 were added in Omnipose, hence the unusual placement in the list.
            # flows = [[o for o in out] for out in zip(rgb, dP, cellprob, p, bd, tr, affinity, bounds)]
            flows = [list(item) for item in
                     zip(flow_RGB, dP, dist, p, bd, tr, affinity, bounds)]  # not sure which is faster of these yet
            return masks, flows, []


        elif (is_list or is_stack) and correct_shape:
            masks, styles, flows = [], [], []

            tqdm_out = utils.TqdmToLogger(models_logger, level=logging.INFO)
            nimg = len(x)
            iterator = trange(nimg, file=tqdm_out, disable=not show_progress) if nimg > 1 else range(nimg)
            # note: ~ is bitwise flip, overloaded to act as elementwise not for numpy arrays
            # but for boolean variables, must use "not" operator isstead
            if verbose:
                models_logger.info('Evaluating one image at a time')

            for i in iterator:
                dia = diameter[i] if isinstance(diameter, list) or isinstance(diameter, np.ndarray) else diameter
                rsc = rescale[i] if isinstance(rescale, list) or isinstance(rescale, np.ndarray) else rescale
                chn = channels if channels is None else channels[i] if (len(channels) == len(x) and
                                                                        (isinstance(channels[i], list)
                                                                         or isinstance(channels[i], np.ndarray)) and
                                                                        len(channels[i]) == 2) else channels

                maski, stylei, flowi = self.eval(x[i],
                                                 batch_size=batch_size,
                                                 channels=chn,
                                                 channel_axis=channel_axis,
                                                 z_axis=z_axis,
                                                 normalize=normalize,
                                                 invert=invert,
                                                 rescale=rsc,
                                                 diameter=dia,
                                                 do_3D=do_3D,
                                                 anisotropy=anisotropy,
                                                 net_avg=net_avg,
                                                 augment=augment,
                                                 tile=tile,
                                                 tile_overlap=tile_overlap,
                                                 bsize=bsize,
                                                 resample=resample,
                                                 interp=interp,
                                                 cluster=cluster,
                                                 suppress=suppress,
                                                 boundary_seg=boundary_seg,
                                                 affinity_seg=affinity_seg,
                                                 despur=despur,
                                                 mask_threshold=mask_threshold,
                                                 diam_threshold=diam_threshold,
                                                 flow_threshold=flow_threshold,
                                                 niter=niter,
                                                 flow_factor=flow_factor,
                                                 compute_masks=compute_masks,
                                                 min_size=min_size,
                                                 max_size=max_size,
                                                 stitch_threshold=stitch_threshold,
                                                 progress=progress,
                                                 show_progress=show_progress,
                                                 omni=omni,
                                                 calc_trace=calc_trace,
                                                 verbose=verbose,
                                                 transparency=transparency,
                                                 loop_run=(i > 0),
                                                 model_loaded=model_loaded)
                masks.append(maski)
                flows.append(flowi)
                styles.append(stylei)
            return masks, styles, flows

        else:
            if not model_loaded and (isinstance(self.pretrained_model, list) and not net_avg and not loop_run):

                # whether or not we are using dataparallel
                if self.torch and self.gpu:
                    models_logger.info(f'using dataparallel')
                    net = self.net.module

                    # if ARM:
                    #     models_logger.info('On ARM, OMP_NUM_THREADS set to 1')
                    #     os.environ['OMP_NUM_THREADS'] = '1'

                else:
                    net = self.net
                    models_logger.info('not using dataparallel')

                if verbose:
                    models_logger.info(f'network initialized.')

                net.load_model(self.pretrained_model[0], cpu=(not self.gpu))
                if not self.torch:
                    net.collect_params().grad_req = 'null'

            if verbose:
                models_logger.info('shape before transforms.convert_image(): {}'.format(x.shape))
                models_logger.info(f'model dim: {self.dim}')

            # This takes care of the special case of grasycale, padding with zeros if the model was trained like that
            x = transforms.convert_image(x, channels, channel_axis=channel_axis, z_axis=z_axis,
                                         do_3D=(do_3D or stitch_threshold > 0), normalize=False,
                                         invert=False, nchan=self.nchan, dim=self.dim, omni=omni)

            if verbose:
                models_logger.info('shape after transforms.convert_image(): {}'.format(x.shape))

            if x.ndim < self.dim + 2:  # we need (nimg, *dims, nchan), so 2D has 4, 3D has 5, etc.
                x = x[np.newaxis]

                if verbose:
                    models_logger.info('shape now {}'.format(x.shape))

            self.batch_size = batch_size
            rescale = self.diam_mean / diameter if (
                        rescale is None and (diameter is not None and diameter > 0)) else rescale
            rescale = 1.0 if rescale is None else rescale

            masks, styles, dP, cellprob, p, bd, tr, affinity, bounds = self._run_cp(x,
                                                                                    compute_masks=compute_masks,
                                                                                    normalize=normalize,
                                                                                    invert=invert,
                                                                                    rescale=rescale,
                                                                                    net_avg=net_avg,
                                                                                    resample=resample,
                                                                                    augment=augment,
                                                                                    tile=tile,
                                                                                    tile_overlap=tile_overlap,
                                                                                    bsize=bsize,
                                                                                    mask_threshold=mask_threshold,
                                                                                    diam_threshold=diam_threshold,
                                                                                    flow_threshold=flow_threshold,
                                                                                    niter=niter,
                                                                                    flow_factor=flow_factor,
                                                                                    interp=interp,
                                                                                    cluster=cluster,
                                                                                    suppress=suppress,
                                                                                    boundary_seg=boundary_seg,
                                                                                    affinity_seg=affinity_seg,
                                                                                    despur=despur,
                                                                                    min_size=min_size,
                                                                                    max_size=max_size,
                                                                                    do_3D=do_3D,
                                                                                    anisotropy=anisotropy,
                                                                                    stitch_threshold=stitch_threshold,
                                                                                    omni=omni,
                                                                                    calc_trace=calc_trace,
                                                                                    show_progress=show_progress,
                                                                                    verbose=verbose)

            # the flow list stores:
            # (1) RGB representation of flows
            # (2) flow components
            # (3) cellprob (cp) or distance field (op)
            # (4) pixel coordinates after Euler integration
            # (5) boundary output (nclasses=4)
            # (6) pixel trajectories during Euler integation (trace=True)
            # (7) augmented affinity graph (coords+affinity) of shape (dim,nstep,npix)
            # (8) binary boundary map

            # 5-8 were added in Omnipose, hence the unusual placement in the list.
            flows = [plot.dx_to_circ(dP, transparency=transparency)
                     if self.nclasses > 1 else np.zeros(cellprob.shape + (3 + transparency,), np.uint8),
                     dP, cellprob, p, bd, tr, affinity, bounds]

            return masks, flows, styles

def steps_batch(p, dP, niter, supr_fac, omni=True, suppress=True, interp=True,
                calc_trace=False, calc_bd=False, verbose=False):
    """Euler integration of pixel locations p subject to flow dP for niter steps in N dimensions.

    Parameters
    ----------------
    p: float32, tensor
        pixel locations [axis x Lz x Ly x Lx] (start at initial meshgrid)
    dP: float32, ND array
        flows [axis x Lz x Ly x Lx]
    niter: int32
        number of iterations of dynamics to run

    Returns
    ---------------
    p: float32, ND array
        final locations of each pixel after dynamics

    """
    align_corners = True
    # mode = 'nearest' if (omni and not suppress) else 'bilinear'

    # we want to use bilinear interpolation if using Euler suppression
    # Affinity reconstruction does not require Euler suppression
    # and we want to also be able to toggle this globally with interp arg
    # (omni and and not suppress) is false when affinity is on
    interp = interp and not suppress
    mode = 'bilinear' if interp else 'nearest'
    if verbose:
        omnipose_logger.info(f'interp is {interp}, interpolation mode is {mode}')

    d = dP.shape[1]  # number of components = number of dimensions
    shape = dP.shape[2:]  # shape of component array is the shape of the ambient volume
    inds = list(range(d))[::-1]  # grid_sample requires a particular ordering

    # print('inds', inds,p.shape, p.min(), p.max())

    device = dP.device  # get the device from dP tensor

    shape = np.array(shape)[inds] - 1.  # dP is d.Ly.Lx, inds flips this to flipped X-1, Y-1, ...
    # print('SHAPE',shape)
    B, D, I = p.shape
    # print('p...',p.shape,inds,shape)
    # pt = p[:,inds].permute(0,2,1).unsqueeze(1).float()
    pt = p[:, inds].permute(0, 2, 1).view([B] + [1] * (D - 1) + [I, D]).float()

    # print('pt_new',pt.shape)

    pt0 = pt.clone()  # save first
    flow = dP[:, inds]  # inds is just flipping the spatial component ordering from TYX to XYT

    # print('point, flow shape',pt.shape,flow.shape)

    for k in range(d):
        pt[..., k] = 2 * pt[..., k] / shape[k] - 1
        flow[:, k] = 2 * flow[:, k] / shape[k]

    if calc_trace:
        dims = [-1, niter] + [-1] * (pt.ndim - 1)
        trace = torch.clone(pt).detach().unsqueeze(1).expand(*dims)  # add time

    if omni and OMNI_INSTALLED and suppress:
        dPt0 = torch.nn.functional.grid_sample(flow, pt, mode=mode, align_corners=align_corners)

    for t in range(niter):
        if calc_trace and t > 0:
            trace[:, t].copy_(pt)

        # print('aa',flow.shape,pt.shape)
        dPt = torch.nn.functional.grid_sample(flow, pt, mode=mode,
                                              align_corners=align_corners)

        if omni and OMNI_INSTALLED and suppress:
            dPt = (dPt + dPt0) / 2.  # average with previous flow
            dPt0.copy_(dPt)  # update old flow
            dPt /= step_factor(t, supr_fac)  # suppression factor

        for k in range(d):  # clamp the final pixel locations
            pt[..., k] = torch.clamp(pt[..., k] + dPt[:, k], -1., 1.)

    pt = (pt + 1) * 0.5
    for k in range(d):
        pt[..., k] *= shape[k]

    if calc_trace:
        trace = (trace + 1) * 0.5
        for k in range(d):
            trace[..., k] *= shape[k]

    if calc_trace:
        # tr =  trace[...,inds].permute(0,1,-1,2,3)
        tr = trace[..., inds].transpose(-1, 1).contiguous()

    else:
        tr = None
    # p =  pt[...,inds].permute(0,-1,1,2)
    p = pt[..., inds].transpose(-1, 1).contiguous()

    empty_cache()
    return p, tr

def step_factor(t, sup_fac):
    """ Euler integration suppression factor.

    Conveneient wrapper function allowed me to test out several supression factors.

    Parameters
    -------------
    t: int
        time step
    """
    return (sup_fac + t)
if __name__ == "__main__":
    a = CellposeModelCustomEuler(nclasses=1)
