class FileMixin:
    """Abstract class, provides interface to work with a file (open, close and context manager)"""

    def __init__(self, path, **kwargs):
        super().__init__(**kwargs)
        self._path = path
        self._descriptor = None
        self._shape = None

    def open(self):
        raise NotImplementedError

    def close(self):
        self._descriptor.close()
        self._descriptor = None
        self._shape = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, traceback):
        self.close()

    def __getitem__(self, item):
        if not self._descriptor:
            raise ValueError(f'File {self._path} is not opened')
        return super().__getitem__(item)

    def __setitem__(self, item, data):
        if not self._descriptor:
            raise ValueError(f'File {self._path} is not opened')
        super().__setitem__(item, data)

    @property
    def shape(self):
        if not self._descriptor:
            raise ValueError(f'File {self._path} is not opened')
        return self._shape
