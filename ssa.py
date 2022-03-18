from typing import List, Tuple, Dict, DefaultDict, Optional, Set
from enum import Enum
from collections import defaultdict
from copy import copy
import warnings


class SSACompileError(Exception):
    pass


class UninitializedVariableWarning(UserWarning):
    pass


class BasicBlockType(Enum):
    FUNC_ROOT = "root"
    FALL_THROUGH = "fall-through"
    BRANCH = "branch"
    IF_JOIN = "if_join"
    WHILE_JOIN = "while_join"


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


no_cse_operations = [
    Operation.READ,
    Operation.WRITE,
    Operation.WRITE_NL,
    Operation.CALL
]

void_operations = [
    Operation.STORE,
    Operation.END,
    Operation.BRA,
    Operation.BNE,
    Operation.BEQ,
    Operation.BLE,
    Operation.BLT,
    Operation.BGE,
    Operation.BGT,
    Operation.WRITE,
    Operation.WRITE_NL
]


class Operand:
    def __str__(self):
        pass

    def __eq__(self, other):
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
        self.register: Optional[int] = None
        self.is_void: bool = False
        self.is_dead: bool = False
        self.instr_op = InstructionOp(self)
        self.operation: Operation = operation
        self.operands: List[Operand] = list(operands)
        self.dominator: Optional[Instruction] = None  # Previous instruction with the same kind

    def __str__(self):
        if self.register:
            if self.register > 32:
                label = f"Virtual R{self.register - 32}"
            else:
                label = f"R{self.register}"
        else:
            label = self.i
        if self.operation is None:
            return f"{label}: <empty>"
        elif self.operation == Operation.CALL:
            return f"{label}: {str(self.operands[0])}"  # Operand is always a FuncCallOp.
        else:
            return f"{label}: {self.operation.value} {' '.join([str(op) for op in self.operands])}"

    def __eq__(self, other):
        return (isinstance(other, Instruction) and
                self.operation == other.operation and
                len(self.operands) == len(other.operands) and
                all([l == r for (l, r) in zip(self.operands, other.operands)]))

    def __hash__(self):
        return self.i


class InstructionOp(Operand):
    def __init__(self, instr: Instruction):
        self.instr = instr

    def __str__(self):
        if self.instr.register:
            if self.instr.register > 32:
                return f"Virtual R{self.instr.register - 32}"
            else:
                return f"R{self.instr.register}"
        else:
            return f"({self.instr.i})"

    def __eq__(self, other):
        return isinstance(other, InstructionOp) and self.instr.i == other.instr.i


class Function:
    def __init__(self, name: str, arg_names: List[str] = None, is_void: bool = True):
        self.name = name
        self.arg_names = arg_names
        self.instr_counter = 0
        self.block_counter = 0  # A function's root block
        self.is_void = is_void
        self.last_block = None

    def get_basic_block_id(self):
        self.block_counter += 1
        return self.block_counter

    def get_instr_id(self):
        self.instr_counter += 1
        return self.instr_counter


class Variable:
    def __init__(self, name: str, dims: Optional[List[int]], operand: Optional[Operand]):
        self.name = name
        self.dims = dims
        self.operand = operand

    def __eq__(self, other):
        return isinstance(other, Variable) and self.name == other.name


class VariableOp(Operand):
    def __init__(self, name: str, operand: Operand):
        self.name = name
        self.operand = operand

    def __eq__(self, other):
        return isinstance(other, VariableOp) and self.name == other.name and self.operand == other.operand

    def __str__(self):
        return f"{self.name}:{self.operand}"


class BasicBlock:
    def __init__(self,
                 func: Function,
                 sym_table: Dict[str, Optional[Variable]],
                 instr_dominators: DefaultDict[Operation, List[Instruction]],
                 basic_block_type: BasicBlockType,
                 unresolved_num_nested_while_loops: int = 0):
        self.backward_pass_visited = False
        self.live_set: Set[Instruction] = set()
        self.type = basic_block_type
        self.consume_unreachable_instrs = False
        self.unresolved_num_nested_while_loops = unresolved_num_nested_while_loops
        self.factor = pow(10, unresolved_num_nested_while_loops)
        self.func: Function = func
        self.basic_block_id: int = func.get_basic_block_id()
        self.instrs: List[Instruction] = []
        self.branch_block: Optional[BasicBlock] = None
        self.fall_through_block: Optional[BasicBlock] = None
        self.preds: List[BasicBlock] = []
        self.dominates: List[BasicBlock] = []
        self.sym_table: Dict[str, Optional[Variable]] = {}
        for ident, var in sym_table.items():
            self.sym_table[ident] = copy(var)
        self.instr_dominators: DefaultDict[Operation, List[Instruction]] = instr_dominators.copy()

    def decl_var(self, ident: str, dims: Optional[List[int]] = None):
        if dims is None:
            self.sym_table[ident] = Variable(ident, dims, UninitializedVarOp(ident, self))
        else:
            self.sym_table[ident] = Variable(ident, dims, VarAddressOp(ident))

    def set_var_op(self, ident: str, operand: Operand) -> None:
        self.sym_table[ident].operand = operand

    def get_var(self, ident: str) -> Variable:
        if ident not in self.sym_table:
            raise Exception(f"{ident} is not declared in {self.func.name}.")

        # Wrap by VariableOp if the variable is nested by while loops and the value can be changed.
        if self.unresolved_num_nested_while_loops > 0 and not isinstance(self.sym_table[ident].operand, VariableOp):
            self.sym_table[ident].operand = VariableOp(ident, self.sym_table[ident].operand)

        return self.sym_table[ident]

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
        join = ""
        if self.type in [BasicBlockType.IF_JOIN, BasicBlockType.WHILE_JOIN]:
            join = f"{self.type.value}\\n"
        live_instr = []
        for instr in self.instrs:
            if not instr.is_dead:
                live_instr.append(instr)
        instrs = "|".join([str(instr) for instr in live_instr])
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
                            f' -> {subgraph_prefix}bb{dom.basic_block_id}:b [color=blue, style=dotted, label="dom"];')

        return dot_block, dot_branches, dot_doms

    def emit(self, operation: Operation, *operands: Operand) -> InstructionOp:
        instr = Instruction(self.func.get_instr_id(), operation, *operands)
        if operation in void_operations:
            instr.is_void = True
        if operation == Operation.CALL:
            func_call_op = operands[0]
            if isinstance(func_call_op, FuncCallOp):
                instr.is_void = func_call_op.func.is_void

        # Common Subexpression Elimination
        if operation not in no_cse_operations:
            for dom_instr in reversed(self.instr_dominators[operation]):
                if dom_instr.operation == Operation.STORE:
                    break  # For load operation, forget everything before the store operation
                if instr == dom_instr:
                    self.func.instr_counter -= 1
                    return dom_instr.instr_op

        # There is no common subexpression -> Add instruction
        if not self.consume_unreachable_instrs:
            self.instrs.append(instr)
            if operation not in [Operation.READ, Operation.WRITE, Operation.WRITE_NL]:
                self.instr_dominators[operation].append(instr)
            if operation == Operation.STORE:
                # Add store operation to the load's tree as well
                self.instr_dominators[Operation.LOAD].append(instr)

        return instr.instr_op


class UninitializedVarOp(Operand):
    def __init__(self, name: str, basic_block: BasicBlock):
        self.name = name
        self.basic_block = basic_block

    def get_default_val_op(self):
        warnings.warn(f"Trying to use variable \033[4m{self.name}\033[0m before initializing. "
                      f"set default value as zero.")
        self.basic_block.set_var_op(self.name, ImmediateOp(0))
        return self.basic_block.get_var(self.name).operand

    def __str__(self):
        return f"#0"

    def __eq__(self, other):
        return (isinstance(other, UninitializedVarOp)
                and self.name == other.name and self.basic_block == other.basic_block)


class SSA:
    def __init__(self):
        self.current_block: Optional[BasicBlock] = None
        self.current_func: Optional[Function] = None
        self.func_roots: Dict[str, BasicBlock] = dict()
        self.builtin_funcs: List[str] = ["InputNum", "OutputNum", "OutputNewLine"]
        # Add built-in functions
        input_num_block = self.get_new_function_block("InputNum", [])
        read_op = input_num_block.emit(Operation.READ)
        input_num_block.emit(Operation.STORE, read_op, VarAddressOp("Memory for Return"))
        input_num_block.emit(Operation.BRA, VarAddressOp("R31(=Return Address)"))

        output_num_block = self.get_new_function_block("OutputNum", ['x'])
        output_num_block.emit(Operation.WRITE, output_num_block.sym_table['x'].operand)
        output_num_block.emit(Operation.BRA, VarAddressOp("R31(=Return Address)"))

        output_newline_block = self.get_new_function_block("OutputNewLine", [])
        output_newline_block.emit(Operation.WRITE_NL)
        output_newline_block.emit(Operation.BRA, VarAddressOp("R31(=Return Address)"))

    def emit(self, operation: Operation, *operands: Operand) -> InstructionOp:
        return self.current_block.emit(operation, *operands)

    def set_current_block(self, block: BasicBlock):
        self.current_block = block
        self.current_func = block.func

    def get_new_function_block(self, name: str, arg_names: List[str], is_void: bool = True) -> BasicBlock:
        """
        Create a new function and a root basic block.
        A function has a separated control flow graph and the new block becomes the root node.
        It automatically set the current function as the new function.
        :param name: Function's name.
        :param arg_names: Function arguments' names.
        :param is_void: True if the function returns nothing, False if the function returns an int.
        :return: Root basic block for the new function.
        """
        func = Function(name, arg_names, is_void)
        func_root_bb = BasicBlock(func, dict(), defaultdict(list), BasicBlockType.FUNC_ROOT, 0)
        self.func_roots[name] = func_root_bb
        self.current_func = func
        self.current_block = func_root_bb
        if arg_names:
            for operand_name in arg_names:
                func_root_bb.decl_var(operand_name)
                func_root_bb.sym_table[operand_name].operand = VarAddressOp(operand_name)
                func_root_bb.sym_table[operand_name].operand = \
                    func_root_bb.emit(Operation.LOAD, func_root_bb.sym_table[operand_name].operand)
        func.last_block = func_root_bb
        return func_root_bb

    def get_new_basic_block(self,
                            basic_block_type: BasicBlockType):
        new_bb = BasicBlock(self.current_func, self.current_block.sym_table,
                            self.current_block.instr_dominators, basic_block_type,
                            self.current_block.unresolved_num_nested_while_loops)
        if basic_block_type == BasicBlockType.FALL_THROUGH:
            self.current_block.fall_through_block = new_bb
        elif basic_block_type == BasicBlockType.BRANCH:
            self.current_block.branch_block = new_bb
        new_bb.preds.append(self.current_block)
        self.current_block.dominates.append(new_bb)
        self.current_func.last_block = new_bb
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
                func_args = func_root_block.func.arg_names
                dot_subgraph.append(f'\t\tlabel="{func_name}({", ".join(func_args)})"')
                for line in func_root_block.dot(i):
                    dot_subgraph.append(f"\t{line}")  # Add indent for each subgraph
                dot_subgraph.append("\t}")
                dot_lines += dot_subgraph

        dot_lines.append('}')

        return '\n'.join(dot_lines)


class FuncCallOp(Operand):
    def __init__(self, ir: SSA, ident: str, *args: Operand):
        if ident not in ir.func_roots:
            raise SSACompileError(f"function {ident} is not declared.")

        self.func = ir.func_roots[ident].func
        func_operands = self.func.arg_names

        if len(func_operands) != len(args):
            raise SSACompileError(f"function {ident}({', '.join(func_operands)})"
                                  f" expects {len(func_operands)} operands, but got {len(args)}.")
        self.ident = ident
        self.args: Tuple[Operand] = args

    def __str__(self):
        if self.ident == Operation.READ:
            return "read"
        elif self.ident == Operation.WRITE:
            return f"write {str(self.args[0])}"
        elif self.ident == Operation.WRITE_NL:
            return "writeNL"
        else:
            return f"{self.ident}({', '.join([str(op) for op in self.args])})"
