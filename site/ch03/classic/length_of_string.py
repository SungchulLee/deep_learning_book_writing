"""
Length of String
"""

def compute_length_of_string_using_recursion(string):
    if string == '':
        return 0
    return 1 + compute_length_of_string_using_recursion(string[1:])


def compute_length_of_string_using_python(string):
    return len(string)


def main():
    # https://edition.cnn.com/2022/04/29/asia/kabul-mosque-explosion-intl/index.html
    strings = """At least 10 people were killed and 30 wounded after an explosion at a mosque in the Afghan capital Kabul after Friday prayers Taliban spokesperson for the chief of police Khalid Zadran said""".split()
    for string in strings:
        result_0 = compute_length_of_string_using_recursion(string)
        result_1 = compute_length_of_string_using_python(string)
        print(f"Computation of length of string '{string}' using recursion : {result_0}")
        print(f"Computation of length of string '{string}' using python    : {result_1}")
        print()
        
        

# === Main ===
if __name__ == "__main__":
    main()
