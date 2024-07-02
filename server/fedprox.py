from server.avg import Server as ServerBase


# FedProx is same as FedAVG. The difference is the value of mu. (0 for FedAvg)
class Server(ServerBase):
    def __init__(self, args):
        super().__init__(args)