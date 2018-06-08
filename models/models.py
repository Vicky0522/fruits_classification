
def create_model(opt):
    model = None
    print(opt.model)
    if opt.model == 'classify':
        assert(opt.dataset_mode == 'classify')
        from .classify_model import ClassifyModel
        model = ClassifyModel()
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
