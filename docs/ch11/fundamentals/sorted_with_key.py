"""
Built-in sorted with key
"""

# ======================================================================

class Employee():
    def __init__(self, name, age, salary):
        self.name = name
        self.age = age
        self.salary = salary

    def __repr__(self):
        return '({}, {}, ${})'.format(self.name, self.age, self.salary)


# === Main ===
if __name__ == "__main__":
    e1 = Employee('Carl', 37, 70000)
    e2 = Employee('Sarah', 29, 80000)
    e3 = Employee('John', 43, 90000)


    employees = [e1, e2, e3]


    if 1: 
        # 이름으로 소팅하기
        def e_sort(emp): 
            return emp.name
    elif 1: 
        # 나이로 소팅하기
        def e_sort(emp):
            return emp.age
    elif 1:
        # 월급으로 소팅하기
        def e_sort(emp):
            return emp.salary


    sorted(employees, key=e_sort)


    sorted(employees, key=e_sort, reverse=True)


    sorted(employees, key=lambda e: e.salary)


    sorted(employees, key=lambda e: e.salary, reverse=True)
