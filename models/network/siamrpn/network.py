from torch import nn


class SiamRPN(nn.Module):
    def __init__(self, backbone, neck, rpn_head):
        super(SiamRPN, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.rpn_head = rpn_head

    def inference_template(self, z):
        zf = self.backbone(z)
        if self.neck is not None:
            zf = self.neck(zf)
        return zf

    def inference_instance(self, zf, x):
        xf = self.backbone(x)
        if self.neck is not None:
            xf = self.neck(xf)
        cls, loc = self.rpn_head(zf, xf)
        return {
                'cls': cls,
                'loc': loc
               }

    def forward(self, data):
        """ only used in training
        """
        template, search = data

        # get feature
        zf = self.backbone(template)
        xf = self.backbone(search)
        if self.neck is not None:
            zf = self.neck(zf)
            xf = self.neck(xf)
        cls, loc = self.rpn_head(zf, xf)
        return cls, loc
