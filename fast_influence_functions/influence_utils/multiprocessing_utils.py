"""Light Modification of the following file (v1.5.1)

   https://github.com/pytorch/pytorch/blob/v1.5.1/torch/multiprocessing/spawn.py

   The major change is to allow device-specific arguments instead of the same
   arguments applied to all processes. The motivation is to reduce sending
   over unnecessary data, which could increase the spawning overhead.
"""

import torch.multiprocessing as torch_mp
from torch.multiprocessing.spawn import (
    _python_version_check, _wrap, multiprocessing, warnings)


# Note: [start_processes]
# mp.start_processes handles both start_method='spawn' and 'fork'. It's supposed to be a
# more generalized API than mp.spawn. Currently we only document mp.spawn as it's the
# CUDA compatible start_method. However, in environments like Ipython notebooks, 'fork'
# works better than 'spawn'. Every helper function we created for mp.spawn is indeed
# general enough, and backends like XLA can reuse them in Colab notebooks as well.
# Currently we only add this API first, we can consider adding it to documentation as
# needed in the future.
def start_processes(fn, list_of_args, nprocs=1, join=True, daemon=False, start_method='spawn'):
    _python_version_check()
    mp = multiprocessing.get_context(start_method)
    error_queues = []
    processes = []
    for i in range(nprocs):
        error_queue = mp.SimpleQueue()
        process = mp.Process(
            target=_wrap,
            args=(fn, i, list_of_args[i], error_queue),
            daemon=daemon,
        )
        process.start()
        error_queues.append(error_queue)
        processes.append(process)

    context = torch_mp.ProcessContext(processes, error_queues)
    if not join:
        return context

    # Loop on join until it returns True or raises an exception.
    while not context.join():
        pass


def spawn(fn, list_of_args, nprocs=1, join=True, daemon=False, start_method='spawn'):
    r"""Spawns ``nprocs`` processes that run ``fn`` with ``args``.
    If one of the processes exits with a non-zero exit status, the
    remaining processes are killed and an exception is raised with the
    cause of termination. In the case an exception was caught in the
    child process, it is forwarded and its traceback is included in
    the exception raised in the parent process.
    Arguments:
        fn (function): Function is called as the entrypoint of the
            spawned process. This function must be defined at the top
            level of a module so it can be pickled and spawned. This
            is a requirement imposed by multiprocessing.
            The function is called as ``fn(i, *args)``, where ``i`` is
            the process index and ``args`` is the passed through tuple
            of arguments.
        args (tuple): Arguments passed to ``fn``.
        nprocs (int): Number of processes to spawn.
        join (bool): Perform a blocking join on all processes.
        daemon (bool): The spawned processes' daemon flag. If set to True,
                       daemonic processes will be created.
        start_method (string): (deprecated) this method will always use ``spawn``
                               as the start method. To use a different start method
                               use ``start_processes()``.
    Returns:
        None if ``join`` is ``True``,
        :class:`~ProcessContext` if ``join`` is ``False``
    """
    if start_method != 'spawn':
        msg = ('This method only supports start_method=spawn (got: %s).\n'
               'To use a different start_method use:\n\t\t'
               ' torch.multiprocessing.start_process(...)' % start_method)
        warnings.warn(msg)
    return start_processes(fn, list_of_args, nprocs, join, daemon, start_method='spawn')
