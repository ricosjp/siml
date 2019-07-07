import chainer as ch


class Classifier(ch.link.Chain):

    def __init__(
            self, predictor, *,
            lossfun=ch.functions.loss.softmax_cross_entropy, accfun=None):
        super().__init__()
        self.lossfun = lossfun
        self.accfun = accfun

        self.y = None
        self.loss = None
        self.accuracy = None

        with self.init_scope():
            self.predictor = predictor

    def __call__(self, x, t):
        self.y = None
        self.loss = None
        self.accuracy = None

        self.y = self.predictor(x)
        self.loss = self.lossfun(self.y, t)
        ch.report({'loss': ch.backends.cuda.to_cpu(self.loss.data)}, self)
        if self.accfun is not None:
            self.accuracy = self.accfun(self.y, t)
            ch.report({'accuracy': self.accuracy}, self)
        return self.loss
