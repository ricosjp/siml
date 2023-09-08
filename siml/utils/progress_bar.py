from tqdm import tqdm


class SimlProgressBar:
    def __init__(self, total: int, desc: str = None) -> None:
        self._trick = 1
        self._total = total
        self._desc = desc
        self._pbar: tqdm = None

    def _create_pbar(self):
        self._pbar = tqdm(
            initial=0,
            leave=False,
            total=self._total,
            desc=self._desc,
            ncols=80,
            ascii=True
        )

    def update(self, trick: int, *, desc: str = None) -> None:
        """Update progress bar

        If Progress bar is not created, create it automatically.
        When counter becomes full, progress bar is destroyed.

        Parameters
        ----------
        trick : int
            Increment to add to the internal counter of iterations
        desc : str, optional
            Set/modify description of the progress bar. [Default: None]
            If None, skip to modify description.
        """
        if self._pbar is None:
            self._create_pbar()

        if desc is not None:
            self._pbar.desc = desc
        self._pbar.update(trick)

        if self._pbar.n == self._total:
            # If counter is full, destroy progress bar completely
            self.destroy()

    def destroy(self) -> None:
        # Unlike tqdm.reset, this method does not show next progress bar.
        self._pbar.close()
        self._pbar = None
