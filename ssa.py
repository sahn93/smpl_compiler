from typing import List, Tuple, Dict, DefaultDict, Optional
from enum import Enum
from collections import defaultdict
from copy import deepcopy
import warnings


class SSACompileError(Exception):
    pass


class UninitializedVariableWarning(UserWarning):
    pass


class BasicBlockType(Enum):
    FUNC_ROOT = "root"
    FALL_THROUGH = "fall-through",
    BRANCH = "branch",
    JOIN = "join"


class Operation(Enum):
    NEG = "neg"
    ADD = "add"
    SUB = "sub"
    MUL = "mul"
    DIV = "div"
    CMP = "cmp"
    ADDA = "adda"
    LOAD = "load"
    STORE = "store"
    PHI = "phi"
    END = "end"
    BRA = "bra"
    BNE = "bne"
    BEQ = "beq"
    BLE = "ble"
    BLT = "blt"
    BGE = "bge"
    BGT = "bgt"
    READ = "read"
    WRITE = "write"
    WRITE_NL = "writeNL"
    CALL = "call"


class Operand:
    pass


class BasicBlockOp(Operand):
    def __init__(self, block_id: int):
        self.block_id = block_id

    def __str__(self):
        return f"BB{self.block_id}"

    def __eq__(self, other):
        return isinstance(other, BasicBlockOp) and self.block_id == other.block_id


class VarAddressOp(Operand):
    def __init__(self, ident: str):
        self.name = ident

    def __str__(self):
        return f"@{self.name}"

    def __eq__(self, other):
        return isinstance(other, VarAddressOp) and self.name == other.name


class ImmediateOp(Operand):
    def __init__(self, value: int):
        self.value = value

    def __str__(self):
        return f"#{self.value}"

    def __eq__(self, other):
        return isinstance(other, ImmediateOp) and self.value == other.value


class Instruction:
    def __init__(self, i: int, operation: Optional[Operation], *operands: Operand):
        self.i: int = i
        self.instr_op = InstructionOp(self)
        self.operation: Operation = operation
        self.operands: List[Operand] = list(operands)
        self.dominator: Optional[Instruction] = None  # Previous instruction with the same kind

    def __str__(self):
        if self.operation is None:
            return f"{self.i}: <empty>"
        elif self.operation == Operation.CALL:
            return f"{self.i}: {str(self.operands[0])}"  # Operand is always a FuncCallOp.
        else:
            return f"{self.i}: {self.operation.value} {' '.join([str(op) for op in self.operands])}"

    def __eq__(self, other):
        return (isinstance(other, Instruction) and
                self.operation == other.operation and
                self.operands == other.operands)


class InstructionOp(Operand):
    def __init__(self, instr: Instruction):
        self.instr = instr

    def __str__(self):
        return f"({self.instr.i})"

    def __eq__(self, other):
        return isinstance(other, InstructionOp) and self.instr == other.instr


class Function:
    def __init__(self, name: str, num_operands: int):
        self.name = name
        self.num_operands = num_operands
        self.instr_counter = 0
        self.block_counter = 0  # A function's root block

    def get_basic_block_id(self):
        self.block_counter += 1
        return self.block_counter

    def get_instr_id(self):
        self.instr_counter += 1
        return self.instr_counter


class Variable:
    def __init__(self, operand: Optional[Operand], dims: Optional[List[int]]):
        self.operand = operand
        self.dims = dims

    def __eq__(self, other):
        return isinstance(other, Variable) and self.operand == other.operand


class BasicBlock:
    def __init__(self,
                 func: Function,
                 sym_table: Dict[str, Optional[Variable]],
                 instr_dominators: DefaultDict[Operation, List[Instruction]],
                 basicblock_type: BasicBlockType):
        self.type = basicblock_type
        self.func: Function = func
        self.basic_block_id: int = func.get_basic_block_id()
        self.instrs: List[Instruction] = []
        self.branch_block: Optional[BasicBlock] = None
        self.fall_through_block: Optional[BasicBlock] = None
        self.dominates: List[BasicBlock] = []
        self.sym_table: Dict[str, Optional[Variable]] = deepcopy(sym_table)
        self.instr_dominators: DefaultDict[Operation, List[Instruction]] = instr_dominators.copy()

    def decl_var(self, ident: str, dims: Optional[List[int]] = None):
        if dims is None:
            self.sym_table[ident] = Variable(None, dims)
        else:
            self.sym_table[ident] = Variable(VarAddressOp(ident), dims)

    def set_var_op(self, ident: str, operand: Operand) -> None:
        self.sym_table[ident].operand = operand

    def get_var_op(self, ident: str) -> Operand:
        if ident not in self.sym_table:
            raise Exception(f"{ident} is not declared in {self.func.name}.")

        if self.sym_table[ident].operand is None:
            warnings.warn(f"{ident} is not initialized."
                          f" Assign an initial value of zero.", UninitializedVariableWarning)
            self.sym_table[ident].operand = ImmediateOp(0)

        return self.sym_table[ident].operand

    def dot(self, subgraph_id: Optional[int] = None) -> List[str]:
        subgraph_prefix = ""
        if subgraph_id is not None:
            subgraph_prefix = f"subgraph{subgraph_id}_"

        bfs_frontiers = [self]
        bfs_visited = set()

        while bfs_frontiers:
            node = bfs_frontiers.pop()
            bfs_visited.add(node)
            if node.branch_block and node.branch_block not in bfs_visited:
                bfs_frontiers.append(node.branch_block)
            if node.fall_through_block and node.fall_through_block not in bfs_visited:
                bfs_frontiers.append(node.fall_through_block)

        dotgraph_blocks = []
        dotgraph_branches = []
        dotgraph_doms = []

        for node in sorted(list(bfs_visited), key=lambda x: x.basic_block_id):
            dot_block, dot_branches, dot_doms = node.dot_node(subgraph_prefix)
            dotgraph_blocks.append(dot_block)
            dotgraph_branches += dot_branches
            dotgraph_doms += dot_doms

        return dotgraph_blocks + [''] + dotgraph_branches + dotgraph_doms

    def dot_node(self, subgraph_prefix: str = ""):
        join = "join\\n" if self.type == BasicBlockType.JOIN else ""
        instrs = "|".join([str(instr) for instr in self.instrs])
        label = f"<b>{join}BB{self.basic_block_id}| {{{instrs}}}"
        dot_block = f'\t{subgraph_prefix}bb{self.basic_block_id} [shape=record, label="{label}"];'

        dot_branches = []

        if self.fall_through_block:
            dot_branches.append(f'{subgraph_prefix}bb{self.basic_block_id}:s'
                                f' -> {subgraph_prefix}bb{self.fall_through_block.basic_block_id}:n'
                                f' [label="fall-through"];')

        if self.branch_block:
            dot_branches.append(f'{subgraph_prefix}bb{self.basic_block_id}:s'
                                f' -> {subgraph_prefix}bb{self.branch_block.basic_block_id}:n'
                                f' [label="branch"];')

        dot_doms = []
        for dom in self.dominates:
            dot_doms.append(f'{subgraph_prefix}bb{self.basic_block_id}:b'
                            f' -> bb{dom.basic_block_id}:b [color=blue, style=dotted, label="dom"];')

        return dot_block, dot_branches, dot_doms

    def emit(self, operation: Optional[Operation], *operands: Operand) -> InstructionOp:
        instr = Instruction(self.func.get_instr_id(), operation, *operands)

        # Common Subexpression Elimination
        if operation is not None:
            for dom_instr in reversed(self.instr_dominators[operation]):
                if dom_instr.operation == Operation.STORE:
                    break  # For load operation to forget everything before the store operation
                if instr == dom_instr:
                    self.func.instr_counter -= 1
                    return dom_instr.instr_op

        # There is no common subexpression.
        self.instrs.append(instr)
        self.instr_dominators[operation].append(instr)
        if operation == Operation.STORE:
            # Add store operation to the load's tree as well
            self.instr_dominators[Operation.LOAD].append(instr)

        return instr.instr_op


class SSA:
    def __init__(self):
        self.current_block: Optional[BasicBlock] = None
        self.current_func: Optional[Function] = None
        self.func_roots: Dict[str, BasicBlock] = dict()
        self.builtin_funcs: List[str] = ["InputNum", "OutputNum", "OutputNewLine"]
        # Add built-in functions
        self.get_new_function_block("InputNum")
        self.emit(Operation.READ)
        self.get_new_function_block("OutputNum", 1)
        self.emit(Operation.WRITE, VarAddressOp("x"))
        self.get_new_function_block("OutputNewLine")
        self.emit(Operation.WRITE_NL)

    def emit(self, operation: Optional[Operation], *operands: Operand) -> InstructionOp:
        return self.current_block.emit(operation, *operands)

    def set_current_block(self, block: BasicBlock):
        self.current_block = block

    def get_new_function_block(self, name: str, num_operands: int = 0) -> BasicBlock:
        """
        Create a new function and a root basic block.
        A function has a separated control flow graph and the new block becomes the root node.
        It automatically set the current function as the new function.
        :param name:
        :param num_operands:
        :return: Root basic block for the new function.
        """
        func = Function(name, num_operands)
        func_root_bb = BasicBlock(func, dict(), defaultdict(list), BasicBlockType.FUNC_ROOT)
        self.func_roots[name] = func_root_bb
        self.current_func = func
        self.current_block = func_root_bb
        return func_root_bb

    def get_new_basicblock(self,
                           basicblock_type: BasicBlockType):
        new_bb = BasicBlock(self.current_func, self.current_block.sym_table,
                            self.current_block.instr_dominators, basicblock_type)
        if basicblock_type == BasicBlockType.FALL_THROUGH:
            self.current_block.fall_through_block = new_bb
        elif basicblock_type == BasicBlockType.BRANCH:
            self.current_block.branch_block = new_bb
        self.current_block.dominates.append(new_bb)
        return new_bb

    def dot(self) -> str:
        dot_lines = ["digraph G {"]

        if len(self.func_roots) == 4:  # There's only main function except the builtin funcs -> no subgraph
            dot_lines += self.func_roots["main"].dot(None)
        else:  # There are more than 0 user define functions -> add functions as subgraphs
            for i, (func_name, func_root_block) in enumerate(self.func_roots.items()):
                if func_name in self.builtin_funcs:
                    continue  # Skip graphs for built-in functions
                dot_subgraph = [f"\tsubgraph cluster_{i} {{"]
                for line in func_root_block.dot(i):
                    dot_subgraph.append(f"\t{line}")  # Add indent for each subgraph
                dot_subgraph.append(f'\t\tlabel="{func_name}"')
                dot_subgraph.append("\t}")
                dot_lines += dot_subgraph

        dot_lines.append('}\n')

        return '\n'.join(dot_lines)


class FuncCallOp(Operand):
    def __init__(self, ir: SSA, ident: str, *operands: Operand):
        if ident not in ir.func_roots:
            raise SSACompileError(f"function {ident} is not declared.")
        if ir.func_roots[ident].func.num_operands != len(operands):
            raise SSACompileError(f"function {ident} expects {ir.func_roots[ident].func.num_operands}"
                                  f" operands, but got {len(operands)}.")

        self.ident = ident
        self.operands: Tuple[Operand] = operands

    def __str__(self):
        if self.ident == Operation.READ:
            return "read"
        elif self.ident == Operation.WRITE:
            return f"write {str(self.operands[0])}"
        elif self.ident == Operation.WRITE_NL:
            return "writeNL"
        else:
            return f"{self.ident}({', '.join([str(op) for op in self.operands])})"
