def divide_num(a, b):
    if b == 0:
        print("Error: Division by zero is not allowed.")
        return None  # or you could raise an exception instead
    else:
        return a / b

# Example usage:
result = divide_num(12, 0)
if result is not None:
    print(result)
else:
    print("Division failed")