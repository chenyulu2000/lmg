class FusionConfigs:
    class AttentionConfigs:
        def __init__(self):
            self.MULTI_HEAD = 8
            self.HIDDEN_SIZE = 1024  # = opt.embed_dim
            self.FF_SIZE = 2048
            self.DROPOUT_R = 0.1

    class FourierConfigs:
        def __init__(self):
            self.HIDDEN_SIZE = 1024  # = opt.embed_dim
            self.DROPOUT_R = 0.1
            self.FF_SIZE = 2048

    class MatrixConfigs:
        def __init__(self):
            self.HIDDEN_SIZE = 1024  # = opt.embed_dim
            self.DROPOUT_R = 0.1
            self.FF_SIZE = 2048

    class ConvolutionConfigs:
        def __init__(self):
            self.HIDDEN_SIZE = 1024  # = opt.embed_dim
            self.DROPOUT_R = 0.1
            self.FF_SIZE = 2048
