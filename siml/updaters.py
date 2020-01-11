


class SimlUpdater(ch.training.updaters.StandardUpdater):

    def update_core(self):
        iterator = self._iterators['main']
        batch = iterator.next()
        in_arrays = self.converter(batch, self.device)

        optimizer = self._optimizers['main']
        if optimizer.target.element_batch_size > 0:
            # Update parameters element batch by element batch
            loss, losses = optimizer.target(**in_arrays)
            for loss_ in losses:
                optimizer.target.cleargrads()
                loss_.backward(loss_scale=optimizer._loss_scale)
                optimizer.update()
        else:
            loss = optimizer.target(**in_arrays)
            optimizer.target.cleargrads()
            loss.backward(loss_scale=optimizer._loss_scale)
            optimizer.update()

        if self.auto_new_epoch and iterator.is_new_epoch:
            optimizer.new_epoch(auto=True)
