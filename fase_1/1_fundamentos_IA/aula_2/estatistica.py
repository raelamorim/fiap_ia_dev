import statistics
import math

nums = [2, 4, 4, 4, 5, 5, 7, 9]

media = statistics.mean(nums)
desvio_padrao = statistics.pstdev(nums) # desvio padrão populacional

desvio_padrao_calculado = math.sqrt(statistics.mean([(n - media) ** 2 for n in nums]))

print(f"A media é: {media:.2f}")
print(f"O desvio padrão é: {desvio_padrao:.2f}")
print(f"O desvio padrão calculado manualmente é: {desvio_padrao_calculado}")

