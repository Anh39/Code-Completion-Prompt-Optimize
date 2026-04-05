from tree_sitter import Language, Parser
import tree_sitter as ts
from typing import TypedDict, cast, Callable, Union, Any
from copy import deepcopy

ENCODING = "utf8"
def decode(data: bytes | None | ts.Node) -> str:
    if isinstance(data, ts.Node):
        return data.text.decode(ENCODING) if data.text else ""
    elif isinstance(data, bytes):
        return data.decode(ENCODING)
    else:
        return ""
PLACE_HOLDER = ("$@$@$", "$X$X$")
def wrap(text: str) -> str:
    result = PLACE_HOLDER[0] + text + PLACE_HOLDER[1]
    return result
def un_wrap(text: str) -> tuple[str, str, str]:
    prefix, rest = text.split(PLACE_HOLDER[0], 1)
    middle, suffix = rest.split(PLACE_HOLDER[1], 1)
    return prefix, middle, suffix
class ParsedNode:
    def __init__(self, encoded: bytes, parent: "ParsedNode | None", node_type: str, start_byte: int, end_byte: int) -> None:
        self._encoded = encoded
        self.parent = parent
        self.type = node_type
        self.start_byte = start_byte
        self.end_byte = end_byte
        self.children: list["ParsedNode"] = []
    # def get_text(self) -> str:
    #     return decode(self._encoded[self.start_byte:self.end_byte])
    def get_compressed_text(self) -> str | None:
        return None
    def get_selective_compressed_text(self, target_node: "ParsedNode") -> str | None:
        if self == target_node:
            return wrap(self.get_full_text())
        else:
            return self.get_compressed_text()
    def get_full_text(self) -> str:
        total_start_byte = self.start_byte
        total_end_byte = self.end_byte
        stack: list[ParsedNode] = [self]
        while len(stack) > 0:
            node = stack.pop(-1)
            total_start_byte = min(node.start_byte, total_start_byte)
            total_end_byte = max(node.end_byte, total_end_byte)
            for child in node.children:
                stack.append(child)
        return decode(self._encoded[self.start_byte:self.end_byte])
    def get_root(self) -> "ParsedNode":
        node = self.parent
        if node is None: return self
        while node.parent != None:
            node = node.parent
        return node
    def is_import(self):
        return self.type in ["import_statement", "import_from_statement"]
    def is_ancestor(self, node: "ParsedNode"):
        node_ = self.parent
        while node_ != None:
            if node_ == node:
                return True
            node_ = node_.parent
        return False
    def is_sibling(self, node: "ParsedNode"):
        if self.parent == None: return False
        for child in self.parent.children:
            if child == node:
                return True
        return False
    def is_class(self):
        return self.type == "class_definition"
    def is_func(self):
        return self.type == "function_definition"
    def leading_indentation(self) -> str:
        start = self.start_byte
        line_start = self._encoded.rfind(b"\n", 0, start) + 1
        indent = self._encoded[line_start:start]
        if indent.strip():
            return ""
        return decode(indent)
    def leading_indentation_byte(self) -> bytes:
        start = self.start_byte
        line_start = self._encoded.rfind(b"\n", 0, start) + 1
        indent = self._encoded[line_start:start]
        if indent.strip():
            return b""
        return indent
    @classmethod
    def parse(cls, text: str, parser: Parser) -> "RootNode":
        encoded = bytes(text, encoding=ENCODING)
        tree = parser.parse(
            encoded
        )
        root = RootNode(encoded, tree.root_node.type)
        # print(root.type)
        # Walk
        def walk(node: ts.Node, parsed_node: ParsedNode):
            for child in node.children:
                parsed_child: ParsedNode
                if child.type == "ERROR":
                    parsed_child = ErrorNode(encoded, parsed_node, child.start_byte, child.end_byte)
                elif child.type == "class_definition":
                    parsed_child = ClassNode(encoded, parsed_node, child.start_byte, child.end_byte)
                elif child.type == "function_definition":
                    parsed_child = FunctionNode(encoded, parsed_node, child.start_byte, child.end_byte)
                elif child.type in ["import_statement", "import_from_statement"]:
                    parsed_child = ImportNode(encoded, parsed_node, node.type, child.start_byte, child.end_byte)
                elif "if __name__ =" in decode(child):
                    parsed_child = SpecialNode(encoded, parsed_node, node.type, child.start_byte, child.end_byte)
                else:
                    parsed_child = ParsedNode(encoded, parsed_node, child.type, child.start_byte, child.end_byte)
                parsed_node.children.append(parsed_child)
                walk(child, parsed_child)
        walk(tree.root_node,  root)
        return root
    def _check_leaf_simple(self):
        return self.type in ["function_definition", "class_definition"]     
class ErrorNode(ParsedNode):
    def __init__(self, encoded: bytes, parent: "ParsedNode | None", start_byte: int, end_byte: int) -> None:
        super().__init__(encoded, parent, "ERROR", start_byte, end_byte)
    def get_compressed_text(self) -> str | None:
        result: list[str] = []
        for child in self.children:
            child_text = child.get_compressed_text()
            # print(child.type)
            if child_text != None:
                result.append(child_text)
        return "\n".join(result)
    def get_selective_compressed_text(self, target_node: ParsedNode):
        if self.get_full_text().strip() == "":
            raise Exception()
        if self == target_node:
            return wrap(self.get_full_text())
        else:
            result: list[str] = []
            for child in self.children:
                child_text = child.get_selective_compressed_text(target_node)
                # print("E", child.type)
                if child_text != None:
                    result.append(child_text)   
                # else:
                #     result.append(child.leading_indentation() + child.get_full_text())
            return "\n".join(result)
class RootNode(ParsedNode):
    def __init__(self, encoded: bytes, node_type: str) -> None:
        super().__init__(encoded, None, node_type, 0, len(encoded))
    def get_compressed_text(self) -> str | None:
        result: list[str] = []
        for child in self.children:
            child_text = child.get_compressed_text()
            # print(child.type)
            if child_text != None:
                result.append(child_text)
        return "\n".join(result)
    def get_selective_compressed_text(self, target_node: ParsedNode):
        if self.get_full_text().strip() == "":
            raise Exception()
        if self == target_node:
            return wrap(self.get_full_text())
        else:
            result: list[str] = []
            for child in self.children:
                child_text = child.get_selective_compressed_text(target_node)
                # print(child.type)
                if child_text != None:
                    result.append(child_text)   
                # else:
                #     result.append(child.leading_indentation() + child.get_full_text())
            return "\n".join(result)
    def find_target(self, byte_boundary: int) -> "ParsedNode":
        last_node: None | ParsedNode = None
        stop = False
        def find_node(node: ParsedNode):
            nonlocal last_node, stop
            if stop: return
            if node.start_byte < byte_boundary:
                last_node = node    
                for child in node.children:
                    find_node(child)
            else:
                stop = True
        find_node(self)
        target = last_node
        if target == None:
            return self
        else:
            target = cast(ParsedNode, target)
            paths: list[ParsedNode] = []
            while target != None:
                paths.append(target)
                target = target.parent
            paths.reverse()
            # Check if inside class first:
            for node in paths:
                if isinstance(node, ClassNode):
                    return cast(ParsedNode, node.parent)
            # Check if inside function:
            for node in paths:
                if isinstance(node, FunctionNode):
                    if isinstance(node.parent, RootNode) or isinstance(node.parent, ErrorNode):
                        return node
            # Check if inside special:
            for node in paths:
                if isinstance(node, SpecialNode):
                    return node
            # Else return root
            return self
class ImportNode(ParsedNode):
    def get_compressed_text(self) -> str | None:
        return self.get_full_text()
class FunctionNode(ParsedNode):
    def __init__(self, encoded: bytes, parent: "ParsedNode | None", start_byte: int, end_byte: int) -> None:
        super().__init__(encoded, parent, "function_definition", start_byte, end_byte)
    def get_compressed_text(self) -> str | None:
        indent = self.leading_indentation_byte()
        body = self._encoded[self.start_byte: self.children[-2].end_byte]
        return decode(indent + body) + " ..."
    def get_selective_compressed_text(self, target_node: "ParsedNode") -> str | None:
        if self == target_node:
            return wrap(self.get_full_text())
        else:
            result = self.get_compressed_text()
            if result != None:
                if not self.is_ancestor(target_node):
                    return result + " ..."
            return result
class ClassNode(ParsedNode):
    def __init__(self, encoded: bytes, parent: "ParsedNode | None", start_byte: int, end_byte: int) -> None:
        super().__init__(encoded, parent, "class_definition", start_byte, end_byte)
    def get_compressed_text(self) -> str | None:
        indent = self.leading_indentation_byte()
        body = self._encoded[self.start_byte: self.children[-2].end_byte]
        text = decode(indent + body)
        result: list[str] = []
        for child in self.children[-1].children:
            child_text = child.get_compressed_text()
            if child_text != None:
                result.append(child_text)
        if len(result) == 0:
            text += " ..."
        result.insert(0, text)
        return "\n".join(result)
    def get_selective_compressed_text(self, target_node: ParsedNode):
        if self == target_node:
            return wrap(self.get_full_text())
        else:
            indent = self.leading_indentation_byte()
            body = self._encoded[self.start_byte: self.children[-2].end_byte]
            text = decode(indent + body)
            result: list[str] = []
            for child in self.children[-1].children:
                child_text = child.get_selective_compressed_text(target_node)
                if child_text != None:
                    result.append(child_text)
                # else:
                #     result.append(child.leading_indentation() + child.get_full_text())
            if len(result) == 0 and not self.is_ancestor(target_node):
                text += " ..."
            result.insert(0, text)
            return "\n".join(result)
class DecoratorNode(ParsedNode):
    def __init__(self, encoded: bytes, parent: "ParsedNode | None", start_byte: int, end_byte: int) -> None:
        super().__init__(encoded, parent, "decorator", start_byte, end_byte)
    def get_compressed_text(self) -> str | None:
        indent = self.leading_indentation_byte()
        body = self._encoded[self.start_byte: self.children[-1].end_byte]
        return decode(indent + body)
class SpecialNode(ParsedNode):
    def get_compressed_text(self) -> str | None:
        total_text = self.get_full_text()
        if "if __name__ =" in total_text:
            return self.leading_indentation() + " ".join([c.get_full_text() for c in self.children[:3]]) + " ..."
        else:
            return None
    def get_selective_compressed_text(self, target_node: "ParsedNode") -> str | None:
        if self == target_node:
            return wrap(self.get_full_text())
        else:
            result = self.get_compressed_text()
            if result != None:
                if not self.is_ancestor(target_node):
                    return result + " ..."
            return result
# First, we handle syntax valid code
# Then, we  handle runtime invalid syntax code (truncate line or some other mechanic)
def collapse_code_valid_synxtax(prefix: str, suffix: str, grammar_path: str) -> tuple[str, str]:
    # We do not handle middle part here since it's already syntax valid and self contained
    total_text = prefix + suffix
    with open("log.py", 'w', encoding='utf-8') as file:
        file.write(total_text)
    byte_boundary = len(bytes(prefix, encoding=ENCODING))
    language = Language(grammar_path, "python")
    parser = Parser()
    parser.set_language(language)
    root = ParsedNode.parse(total_text, parser)
    # print(root.type)
    target_node = root.find_target(byte_boundary)
    # print(type(target_node))
    # print(target_node.get_full_text())
    full_text = root.get_selective_compressed_text(target_node)
    parts = un_wrap(full_text)
    encoded = bytes(parts[1], encoding=ENCODING)
    byte_offset_middle = byte_boundary - target_node.start_byte
    previous_middle = decode(encoded[:byte_offset_middle])
    after_middle = decode(encoded[byte_offset_middle:])
    # print(after_middle)
    return (parts[0]+previous_middle,after_middle+parts[2])
def collapse_code(text: str, grammar_path: str) -> str:
    language = Language(grammar_path, "python")
    parser = Parser()
    parser.set_language(language)
    root = ParsedNode.parse(text, parser)
    result = root.get_compressed_text()
    return result or ""