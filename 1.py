def diccionario_cuadrados(n):
    diccionario = {}
    for i in range(1, n+1):
        diccionario[i] = i**2
        return diccionario


n = int(input("ingresa el numero para crear el diccionario: "))
print(diccionario_cuadrados(n))
#adrian yepes
#kenner pacheco
#haminson bolivar
#christian barros