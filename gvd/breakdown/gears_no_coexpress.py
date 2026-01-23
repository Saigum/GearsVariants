import torch
from gears.model import GEARS_Model # Assuming gears.model is available

class GEARS_No_Coexpress(GEARS_Model):
    def __init__(self, args):
        super().__init__(args)
        self.layers_emb_pos = torch.nn.ModuleList() # Empty module list