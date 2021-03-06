### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import torch
def create_model(opt):
    if opt.model == 'pix2pixHD':
        from .pix2pixHD_model import Pix2PixHDModel, InferenceModel
        if opt.isTrain:
            model = Pix2PixHDModel()
        else:
            model = InferenceModel()
    elif opt.model == 'base1':
        from .Lmark2RGB_model import Lmark2RGBModel1, InferenceModel1
        if opt.isTrain:
            model = Lmark2RGBModel1()
        else:
           model = InferenceModel1()
    elif opt.model == 'base2':
        from .Lmark2RGB_model import Lmark2RGBModel2, InferenceModel2
        if opt.isTrain:
            model = Lmark2RGBModel2()
        else:
            model = InferenceModel2()
    else:
    	from .ui_model import UIModel
    	model = UIModel()
    model.initialize(opt)
    if opt.verbose:
        print("model [%s] was created" % (model.name()))
    if opt.isTrain and len(opt.gpu_ids):
        model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids).cuda()
    return model
