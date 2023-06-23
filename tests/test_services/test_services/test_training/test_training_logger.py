import pytest

from siml.services.training import LogRecordItems


@pytest.mark.parametrize(
    "epoch, train_loss, validation_loss, train_other_losses, validation_other_losses, elapsed_time",  # NOQA
    [
        (12, 0.3455, 0.1223, {"sample": 2.1234}, {"sample": 3.456}, 120)
    ]
)
def test__can_create_log_record_items(
    epoch,
    train_loss,
    validation_loss,
    train_other_losses,
    validation_other_losses,
    elapsed_time
):
    _ = LogRecordItems(
        epoch=epoch,
        train_loss=train_loss,
        validation_loss=validation_loss,
        train_other_losses=train_other_losses,
        validation_other_losses=validation_other_losses,
        elapsed_time=elapsed_time
    )
