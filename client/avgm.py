from client.avg import LocalUpdate as LocalBase

class LocalUpdate(LocalBase):
    def __init__(self, args, dataset, logger):
        super().__init__(args, dataset, logger)
    #  1e-1-u-0.5 for iid; 5e-3-u-0.5 for 0.5_0.5; 1e-2-u-0.5 for 0_0; 5e-3-u-0.3 for 1_1
    