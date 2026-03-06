"""
Dynamic Arrays
"""



# === Main ===
if __name__ == "__main__":
    expenses = [2200, 2350, 2600, 2130, 2190]
    print(expenses)


    # 1 In Feb, how many dollars you spent extra compare to January?
    print(expenses[1] - expenses[0])


    # 2 Find out your total expense 
    # in first quarter (first three months) of the year.
    print(sum(expenses[:3]))


    # 3 Find out if you spent exactly 2000 dollars in any month
    print(2000 in expenses)


    # 4 June month just finished and your expense is 1980 dollar. 
    # Add this item to our monthly expense list
    expenses.append(1980)
    print(expenses)


    # 5 You returned an item that you bought in a month of April and
    # got a refund of 200$. Make a correction to your monthly expense list
    # based on this
    expenses[3] -= 200 
    print(expenses)


    heros = ['spider man','thor','hulk','iron man','captain america']
    print(heros)


    # 1 Length of the list
    print(len(heros))


    # 2 Add 'black panther' at the end of this list
    heros.append('black panther')
    print(heros)


    # 3 You realize that you need to add 'black panther' after 'hulk', 
    # so remove it from the list first and then add it after 'hulk'
    heros.pop()
    idx = heros.index('hulk')
    heros.insert(idx+1, 'black panther')
    print(heros)


    # 4 Now you don't like thor and hulk because they get angry easily :) 
    # So you want to remove thor and hulk from list and 
    # replace them with doctor strange (because he is cool). 
    # Do that with one line of code.
    heros = ['doctor strange' if (hero == 'thor' or hero == 'hulk') 
             else hero
             for hero in heros]
    print(heros)


    # 5 Sort the heros list in alphabetical order 
    # (Hint. Use dir() functions to list down all functions available in list)
    heros.sort()
    print(heros)


    max_num = int(input('Type some posive number : '))
    lst = [i for i in range(1, max_num+1) if i%2==1]
    print(lst)
