# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import functools
from termcolor import colored
import logging
import os
import sys
from typing import Optional

import chibooks.distributed as chi_distributed
from .helpers import MetricLogger, SmoothedValue

# So that calling _configure_logger multiple times won't add many handlers
@functools.lru_cache()
def _configure_logger(
    name: Optional[str] = None,
    *,
    level: int = logging.DEBUG,
    output: Optional[str] = None,
    main_dump_only: bool = True
):
    """
    Configure a logger.

    Adapted from Detectron2.

    Args:
        name: The name of the logger to configure.
        level: The logging level to use.
        output: A file name or a directory to save log. If None, will not save log file.
            If ends with ".txt" or ".log", assumed to be a file name.
            Otherwise, logs will be saved to `output/log.txt`.

    Returns:
        The configured logger.
    """

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    

    # create formatter
    fmt = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'
    # color_fmt = colored('[%(asctime)s %(name)s]', 'green') + \
    #     colored('(%(filename)s %(lineno)d)', 'light_magenta') + \
    #     ': %(levelname)s %(message)s'
    color_fmt = colored('[%(asctime)s %(name)s] ', 'green',) + \
        '%(message)s'
    datefmt = "%Y%m%d %H:%M:%S"
    formatter = logging.Formatter(fmt=color_fmt, datefmt=datefmt)

    # stdout logging for main worker only
    if chi_distributed.is_main_process():
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # file logging for all workers
    if output and (not main_dump_only or chi_distributed.is_main_process()):
        
        global_rank = chi_distributed.get_rank()
        filename = output + "rank{}.log".format(global_rank)
        handler = logging.StreamHandler(open(filename, "a"))
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def setup_logging(
    output: Optional[str] = None,
    *,
    name: Optional[str] = None,
    level: int = logging.DEBUG,
    capture_warnings: bool = True,
    main_dump_only: bool = True
):
    """
    Setup logging.

    Args:
        output: A file name or a directory to save log files. If None, log
            files will not be saved. If output ends with ".txt" or ".log", it
            is assumed to be a file name.
            Otherwise, logs will be saved to `output/log.txt`.
        name: The name of the logger to configure, by default the root logger.
        level: The logging level to use.
        capture_warnings: Whether warnings should be captured as logs.
    """
    logging.captureWarnings(capture_warnings)
    logger = _configure_logger(name, level=level, output=output, main_dump_only=main_dump_only)
    return logger


