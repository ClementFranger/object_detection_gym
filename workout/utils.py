class Schema:

    @classmethod
    def keys(cls):
        return [k for k in vars(cls) if not k.startswith('__')]

    @classmethod
    def values(cls):
        return [getattr(cls, k) for k in cls.keys()]
