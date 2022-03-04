from typing import List, Tuple, Dict
from enum import Enum
import numpy as np


class SSAValueType(Enum):
    IMMEDIATE = 1,
    REGISTER = 2


class SSAVariableType(Enum):
    INT = 1,
    ARRAY = 2


class SSAValue:
    def __init__(self, operand_type: SSAValueType, value: int):
        self.operand_type = operand_type
        self.value = value


class SSAVariable:
    def __init__(self, dim):
        self.value = None
        if not dim:
            self.var_type = SSAVariableType.INT
        else:
            self.var_type = SSAVariableType.ARRAY
            self.dim = dim
            self.value = np.empty(dim)


class SSAInstruction:
    def __init__(self, index: int, opcode: str, operands: Tuple[SSAValue]):
        self.index: int = index
        self.opcode: str = opcode
        self.operands: Tuple[SSAValue] = operands


class BasicBlock:
    def __init__(self, immediate_dom=None):
        self.is_processed = False
        self.instrs: List[SSAInstruction] = []
        self.preds: List[BasicBlock] = []
        self.succs: List[BasicBlock] = []
        self.dominates: List[BasicBlock] = []
        self.sym_table: Dict[SSAVariable] = {}
        if immediate_dom:
            self.sym_table = dict(immediate_dom.sym_table)
