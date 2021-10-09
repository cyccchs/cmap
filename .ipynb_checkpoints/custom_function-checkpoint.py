import torch
import torch.nn as nn

def extract_glimpse(img_batch, size, offsets, centered=True, normalized=True):
    W, H = img_batch.size(-1), img_batch.size(-2)

    if normalized and centered:
        offsets = (offsets+1) * offsets.new_tensor([W/2,H/2])
    elif normalized:
        offsets = offsets * offsets.new_tensor([W,H])
    elif centered:
        raise ValueError(f'Invalid parameter that offsets centered but now normalized')

    h, w = size
    xs = torch.arange(0, w, dtype=img_batch.dtype, device=img_batch.device) - (w-1)/2.0
    ys = torch.arange(0, h, dtype=img_batch.dtype, device=img_batch.device) - (h-1)/2.0

    vy, vx = torch.meshgrid(ys, xs)
    grid = torch.stack([vx, vy], dim=-1)

    offsets_grid = offsets[:,None, None, :] + grid[None, ...]
    offsets_grid = (
            offsets_grid - offsets_grid.new_tensor([W/2,H/2])) / offsets_grid.new_tensor([W/2,H/2])

    return torch.nn.functional.grid_sample(
            img_batch, offsets_grid, mode='bilinear', align_corners=False, padding_mode='zeros')

def extract_multiple_glimpse(img_batch, size, offsets, centered=True, normalized=True):
    patches = []

    for i in range(offsets.size(-2)):
        patch = extract_glimpse(
                img_batch, size, offsets[:, i, :], centered, normalized)
        patches.append(patch)

    return torch.stack(patches, dim=1)
