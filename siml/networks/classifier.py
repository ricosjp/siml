import chainer as ch


class Classifier(ch.link.Chain):

    def __init__(
            self, predictor, *,
            lossfun=ch.functions.loss.softmax_cross_entropy, accfun=None,
            element_batch_size=-1):
        super().__init__()
        self.lossfun = lossfun
        self.accfun = accfun
        self.element_batch_size = element_batch_size

        self.y = None
        self.loss = None
        self.accuracy = None

        with self.init_scope():
            self.predictor = predictor

    def __call__(self, x, t, supports=None):
        self.y = None
        self.loss = None
        self.accuracy = None

        self.y = self.predictor(x, supports)
        self.loss = self.lossfun(self.y, t)
        ch.report({'loss': ch.backends.cuda.to_cpu(self.loss.data)}, self)
        if self.accfun is not None:
            self.accuracy = self.accfun(self.y, t)
            ch.report({'accuracy': self.accuracy}, self)

        if self.element_batch_size > 0:
            length = self.y.shape[-2]
            if self.element_batch_size >= length:
                indices = 1  # No split
            else:
                indices = range(
                    self.element_batch_size, length, self.element_batch_size)

            split_ys = ch.functions.split_axis(
                self.y, indices, axis=-2)
            split_ts = ch.functions.split_axis(
                t, indices, axis=-2)

            losses = [
                self.lossfun(split_y, split_t)
                for split_y, split_t in zip(split_ys, split_ts)]
            return self.loss, losses

        return self.loss
