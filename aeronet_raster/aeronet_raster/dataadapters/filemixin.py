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

    @property
    def shape(self):
        if not self._descriptor:
            raise ValueError(f'File {self._path} is not opened')
        return self._shape
