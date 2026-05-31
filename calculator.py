def add(a, b):
    return a + b


def subtract(a, b):
    return a - b


def main():
    print("加减法计算器")
    print("输入 'q' 退出\n")

    while True:
        expr = input("请输入表达式 (例如: 3 + 5 或 10 - 4): ").strip()
        if expr.lower() == 'q':
            print("再见!")
            break

        if '+' in expr:
            parts = expr.split('+', 1)
            op = '+'
        elif '-' in expr:
            parts = expr.split('-', 1)
            op = '-'
        else:
            print("错误: 请使用 + 或 - 运算符")
            continue

        try:
            a = float(parts[0].strip())
            b = float(parts[1].strip())
        except ValueError:
            print("错误: 请输入有效数字")
            continue

        result = add(a, b) if op == '+' else subtract(a, b)
        print(f"结果: {result}\n")


if __name__ == '__main__':
    main()
