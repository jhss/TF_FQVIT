# Copyright (c) MEGVII Inc. and its affiliates. All Rights Reserved.
from .minmax import MinmaxObserver
from .ptf import PtfObserver

str2observer = {
    'minmax': MinmaxObserver,
    'ptf': PtfObserver
}

def build_observer(observer_str, module_type, bit_type, calibration_mode):
    observer = str2observer[observer_str]
    return observer(module_type, bit_type, calibration_mode)