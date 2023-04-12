from torch.optim.lr_scheduler import LambdaLR


# =======restart the linear warmup strategy with linear warmup==========
def get_restart_linear_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, last_epoch=-1, restart_warmup_steps=0, restart_steps=0
):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
    a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training, will be modified. 
        restart_warmup_steps: 
            the restart_warmup_steps should be set last_epoch + restart_warmup_steps; 

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step: int):

        if (
            restart_steps != 0
            and restart_warmup_steps != 0
            and current_step < restart_steps + restart_warmup_steps
            and current_step >= restart_steps
        ):
            assert current_step >= restart_steps

            # pre-warmup + restart-warmup
            if current_step < num_warmup_steps:
                return (
                    float(current_step - restart_steps)
                    / float(max(1, restart_warmup_steps))
                    * float(restart_steps + restart_warmup_steps)
                    / float(max(1, num_warmup_steps))
                )
            else:
                return (
                    float(current_step - restart_steps)
                    / float(max(1, restart_warmup_steps))
                    * float(num_training_steps - restart_steps - restart_warmup_steps)
                    / float(max(1, num_training_steps - num_warmup_steps))
                )

        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)
