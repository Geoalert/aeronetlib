from aeronet_raster.aeronet_raster.utils.samplers.gridsampler import GridSampler, make_grid, get_safe_shape

# Having image with 3 channels, height=7 and width=8
# We want to iterate it over 2d grid (over all channels at once) with stride 2
# since height is not divisible by stride, safe height must be bigger (8)
shape = (3, 7, 8)
stride = (3, 2, 2)

safe_shape = get_safe_shape(shape, stride)
print(safe_shape)

sampler = GridSampler(make_grid([(0, ss) for ss in safe_shape], stride))
for s in sampler:
    print(s)
