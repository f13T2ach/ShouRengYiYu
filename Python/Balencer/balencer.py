import numpy as np
from sympy import Matrix, lcm
import re

# 已知元素表（118 个已知化学元素）
KNOWN_ELEMENTS = {
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
    "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
    "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"
}

def parse_compound(compound):
    """解析化学式，支持多层括号，返回元素及其计数"""
    def expand(match):
        group, multiplier = match.groups()
        expanded = ''.join(f'{atom}{int(count) * int(multiplier)}' if count else f'{atom}{multiplier}'
                           for atom, count in re.findall(r'([A-Z][a-z]*)(\d*)', group))
        return expanded

    while '(' in compound:
        compound = re.sub(r'\(([^()]+)\)(\d+)', expand, compound)

    elements = re.findall(r'([A-Z][a-z]*)(\d*)', compound)
    element_dict = {element: int(count) if count else 1 for element, count in elements}

    # 验证元素是否合法
    for element in element_dict.keys():
        if element not in KNOWN_ELEMENTS:
            raise ValueError(f"未知元素：{element}")

    return element_dict

def get_element_matrix(reactants, products):
    """构建化学方程式的矩阵"""
    all_compounds = reactants + products
    elements = sorted(set().union(*[parse_compound(c).keys() for c in all_compounds]))
    
    matrix = []
    for compound in all_compounds:
        compound_dict = parse_compound(compound)
        row = [compound_dict.get(e, 0) for e in elements]
        matrix.append(row)
    
    return np.array(matrix).T, len(reactants), elements

def balance_equation(reactants, products):
    """配平化学方程式"""
    matrix, split, elements = get_element_matrix(reactants, products)
    
    # 左右两边符号相反
    augmented_matrix = np.hstack([matrix[:, :split], -matrix[:, split:]])
    
    # 转化为符号矩阵并求解
    sympy_matrix = Matrix(augmented_matrix)
    nullspace = sympy_matrix.nullspace()
    if not nullspace:
        raise ValueError("配平化学方程式失败 矩阵无解")
    
    coeffs = nullspace[0]
    lcm_value = lcm([item.q for item in coeffs])
    coeffs = [(item * lcm_value).simplify() for item in coeffs]
    
    # 转为整数系数
    coeffs = [int(c) for c in coeffs]
    reactant_coeffs = coeffs[:split]
    product_coeffs = coeffs[split:]
    return reactant_coeffs, product_coeffs

def format_equation(reactants, products, reactant_coeffs, product_coeffs):
    """格式化输出化学方程式"""
    reactant_part = " + ".join(
        f"{coef if coef > 1 else ''}{reactant}" for coef, reactant in zip(reactant_coeffs, reactants)
    )
    product_part = " + ".join(
        f"{coef if coef > 1 else ''}{product}" for coef, product in zip(product_coeffs, products)
    )
    return f"{reactant_part} -> {product_part}"

def main():
    print("反应物（空格分隔，如 H2 O2）：")
    reactants = input().strip().split()
    print("生成物（空格分隔，如 H2O）：")
    products = input().strip().split()

    try:
        reactant_coeffs, product_coeffs = balance_equation(reactants, products)
        equation = format_equation(reactants, products, reactant_coeffs, product_coeffs)
        print("配平后的化学方程式：", equation)
    except ValueError as e:
        print(e)

if __name__ == "__main__":
    main()
