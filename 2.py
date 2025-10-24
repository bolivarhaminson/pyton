def contar_caracteres(cadena): 
    contador = {}  
    for caracter in cadena: 
        if caracter in contador: 
            contador[caracter] += 1  
        else:
            contador[caracter] = 1 
    return contador
      
texto = input("introduce una cadena:")
resultado= contar_caracteres
print("Cantidad de apariciones de cada carácter:") 
print(resultado)
#adrian yepes
#kenner pacheco
#haminson bolivar
#christian barros