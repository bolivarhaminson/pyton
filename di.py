respuesta = str(input("¿Quieres comprar frutas? ")).lower()

dicF = {
    "pera": 3000,
    "manzana": 4000,
    "mango": 3500
}

total_compra = 0  

if respuesta == "si": 
    while respuesta == "si":
        fruta = input("Ingresa el nombre de la fruta: ").lower()
        cantidad = int(input(f"Ingrese la cantidad de {fruta}s: "))

        if fruta in dicF:
            total = dicF[fruta] * cantidad
            total_compra += total  
            print(f"El total por {cantidad} {fruta}(s) es: ${total}")
        else:
            print(f"La fruta '{fruta}' no está disponible.")

        respuesta = input("¿Quieres comprar otra fruta? ").lower()

    
    print(f"El total de tu compra es: ${total_compra}")
else:
    print("Gracias por visitarnos.")

#adrian yepes
#kenner pacheco
#haminson bolivar
#christian barros