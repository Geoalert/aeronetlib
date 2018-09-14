# def parse_filepath(fp):
#     return os.path.basename(fp).split('.')[0]
#
#
# def get_band_filepathes(directory, band_names, extensions):
#     files = os.listdir(directory)
#     files = [x for x in files if x.split('.')[-1] in extensions]
#
#     if band_names is not None:
#         files = [x for x in files if x.split('.')[0] in band_names]
#     return [os.path.join(directory, f) for f in files]
#
#
# def decode(s: str, shape: tuple, dtype: str):
#     return np.frombuffer(base64.decodebytes(s), dtype=dtype).reshape(shape)
#
#
# def encode(a: 'np.array'):
#     return base64.b64encode(a)
#
#
#
# from rasterio.windows import Window
#
# def to_float(window: Window, x_max, y_max):
#     w = window
#     new_window = [w.col_off / x_max, w.row_off / y_max,
#                   w.width / x_max, w.height / y_max]
#     return Window(*new_window)
#
# def to_int(window: Window, x_max, y_max):
#     w = window
#     new_window = [w.col_off * x_max, w.row_off * y_max,
#                   w.width * x_max, w.height * y_max]
#     new_window = list(map(round, new_window))
#     return Window(*new_window)
#
# # class Window:
# #
# #     def __init__(self, y, x, height, width):
# #         self.x = x
# #         self.y = y
# #         self.width = width
# #         self.height = height
# #
# #     def __repr__(self):
# #         return str((self.y, self.x, self.height, self.width))
# #
# #     @property
# #     def as_slice(self):
# #         return (self.y, self.y + self.height, self.x, self.x + self.height)
# #
# #     @property
# #     def as_bbox(self):
# #         return (self.y, self.x, self.y + self.heigth, self.x + self.width)
# #
# #     def as_coords(self, transform: Affine):
# #         x1 = transform.c + transform.a * self.x
# #         x2 = x1 + self.width * transform.a
# #
# #         y1 = transform.f + transform.e * self.y
# #         y2 = y1 + self.height * transform.e
# #
# #         return [y1, x1, y2, x2]
# #
# #     def to_transfrom(self, transform: Affine):
# #         new_x = transform.c + transform.a * self.x
# #         new_y = transform.f + transform.e * self.y
# #
# #         new_transform = Affine(transform.a, transform.b, new_x,
# #                                transform.d, transform.e, new_y)
# #
# #         return new_transform
#
#     # @classmethod
#     # def read_window(cls, fp, window: Window):
#     #
#     #     name = parse_filepath(fp)
#     #
#     #     with rasterio.open(fp) as src:
#     #         crs = src.crs
#     #         transform = src.transform
#     #         nodata = src.nodata
#     #         shape = src.shape
#     #         window = to_int(window, shape[1], shape[0])
#     #         raster = src.read(window=window)
#     #
#     #     new_x = transform.c + window.col_off * transform.a
#     #     new_y = transform.f + window.row_off * transform.e
#     #     transform = Affine(transform.a, transform.b, new_x,
#     #                        transform.d, transform.e, new_y)
#     #
#     #     band = cls(name, raster, crs, transform, nodata)
#     #     return band
#
#
# #
# # def _get_paddings(self, window, bound):
# #
# #     x, y, w, h = window
# #
# #     padding_left = abs(min(0, x - bound))
# #     padding_top = abs(min(0, y - bound))
# #
# #     padding_right = abs(min(0, self.width - (x + w + bound)))
# #     padding_bottom = abs(min(0, self.height - (y + h + bound)))
# #
# #     reading_window = (
# #         (x - bound + padding_left,
# #          x + w + bound - padding_right),
# #
# #         (y - bound + padding_top,
# #          y + h + bound - padding_bottom),
# #     )
# #
# #     paddings = (
# #         (0, 0),
# #         (padding_top, padding_bottom),
# #         (padding_left, padding_right),
# #     )
# #
# #     return reading_window, paddings
# #
# #
# # def bounded_sample(self, x, y, width, height, padding=True, bounds=0):
# #
# #     w, p = self._get_paddings((x, y, width, height), 512)
# #     print(w, p)
# #     sample = self.
# #     dst_raster = self.band.read(window=((x, x+width), (y, y+height)))
# #
# #     sample = Sample(dst_name, dst_raster, dst_crs, dst_transform, dst_nodata)
# #
# #     return sample