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
      # Different parenthesis placements
      expressions = [
          f"((({num_permutation[0]} {operator_symbols[op_combination[0]]} {num_permutation[1]}) {operator_symbols[op_combination[1]]} {num_permutation[2]}) {operator_symbols[op_combination[2]]} {num_permutation[3]})",
          f"(({num_permutation[0]} {operator_symbols[op_combination[0]]} {num_permutation[1]}) {operator_symbols[op_combination[1]]} ({num_permutation[2]} {operator_symbols[op_combination[2]]} {num_permutation[3]}))",
          f"({num_permutation[0]} {operator_symbols[op_combination[0]]} ({num_permutation[1]} {operator_symbols[op_combination[1]]} ({num_permutation[2]} {operator_symbols[op_combination[2]]} {num_permutation[3]})))",
          f"({num_permutation[0]} {operator_symbols[op_combination[0]]} (({num_permutation[1]} {operator_symbols[op_combination[1]]} {num_permutation[2]}) {operator_symbols[op_combination[2]]} {num_permutation[3]}))",
          f"({num_permutation[0]} {operator_symbols[op_combination[0]]} {num_permutation[1]}) {operator_symbols[op_combination[1]]} ({num_permutation[2]} {operator_symbols[op_combination[2]]} {num_permutation[3]})",
      ]
      for expression in expressions:
        try:
          result = eval(expression)
          if abs(result - 24) < 1e-6:
            print(f"Solution {solutions_found+1}: {expression}")
            solutions_found += 1
            break  # Move on to the next operator combination if a solution is found for this permutation
        except ZeroDivisionError:
          pass
  if solutions_found > 0:
    print(f"A total of {solutions_found} solutions were found.")
    return True
  else:
    print("No solutions were found.")
    return False

numbers = [4, 3, 7, 8]
find_24(numbers)