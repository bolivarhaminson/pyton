# Productos y Precios
productos = []
precios = []

for i in range(5):
  nombre = input("Ingrese el nombre del producto: ")
  precio = float(input("Ingrese el precio del producto: "))

  productos.append(nombre)
  precios.append(precio)

total = sum(precios)

mas_caro_precio = max(precios)
mas_barato_precio = min(precios)

# Obtener los nombres de los productos más caro y más barato
indice_mas_caro = precios.index(mas_caro_precio)
nombre_mas_caro = productos[indice_mas_caro]

indice_mas_barato = precios.index(mas_barato_precio)
nombre_mas_barato = productos[indice_mas_barato]

print("Productos:", productos)
print("Precios:", precios)
print("Total de precios:", total)
print("Producto más caro:", nombre_mas_caro, "(", mas_caro_precio, ")")
print("Producto más barato:", nombre_mas_barato, "(", mas_barato_precio, ")")

#Haminson Bolivar
#Christian Barros
