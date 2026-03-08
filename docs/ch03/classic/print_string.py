"""
Print String
"""

def print_string_using_recursion(string):
    if string == '':
        return 
    print(string[0], end='')
    print_string_using_recursion(string[1:])


def print_string_using_using_python(string):
    print(string)


def main():
    # https://edition.cnn.com/2022/04/29/asia/kabul-mosque-explosion-intl/index.html
    string = """At least 10 people were killed and 30 wounded after an explosion at a mosque in the Afghan capital Kabul after Friday prayers Taliban spokesperson for the chief of police Khalid Zadran said"""
    print(f"Print string using recursion :")
    print_string_using_recursion(string)
    print()
    print()
    print(f"Print string using python built-in functionality :")
    print_string_using_using_python(string)
        

# === Main ===
if __name__ == "__main__":
    main()
