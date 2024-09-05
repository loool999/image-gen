import itertools
from operator import add, sub, mul, truediv

def calculate(numbers, operations):
  try:
    result = numbers[0]
    for i in range(3):
      result = operations[i](result, numbers[i+1])
    return result
  except ZeroDivisionError:
    return None

def find_24(numbers):
  operators = [add, sub, mul, truediv]
  operator_symbols = {add: "+", sub: "-", mul: "*", truediv: "/"}
  solutions_found = 0
  for num_permutation in itertools.permutations(numbers):
    for op_combination in itertools.product(operators, repeat=3):
      result = calculate(num_permutation, op_combination)
      if result is not None and abs(result - 24) < 1e-6:
        a, b, c, d = num_permutation
        op1, op2, op3 = op_combination
        expression = f"((({a} {operator_symbols[op1]} {b}) {operator_symbols[op2]} {c}) {operator_symbols[op3]} {d})"
        print(f"Solution {solutions_found+1}: {expression}")
        solutions_found += 1
  if solutions_found > 0:
    print(f"A total of {solutions_found} solutions were found.")
    return True
  else:
    print("No solutions were found.")
    return False

numbers = [2, 2, 7, 10]
find_24(numbers)