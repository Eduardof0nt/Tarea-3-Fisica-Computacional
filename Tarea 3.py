import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack

Tm = 0.0001 #Tiempo de muestreo
frec_filt = 1000 #Frecuencia de filtrado de 1000 Hz

tiempo = np.arange(0.0,0.05,Tm) #Se crea un arreglo que contiene todos los pasos discretos de tiempo de la señal que se va a generar.

#Se crea la señal que consiste de una onda senoidal con frecuencia de 440 Hz (un La en la cuarta octava del piano) y amplitud de 5 y otra con una frecuencia de 523.25 Hz (un Do en la quinta octava del piano) y amplitud de 3.
#También se le añade ruido usando la función random.normal de NumPy, este con una media de 0, una desviación estandar de 0.1 y una ampitud de 2.
señal = 5*np.sin(2*np.pi*440*tiempo) + 3*np.sin(2*scipy.pi*523.25*tiempo) + 2*np.random.normal(0, 0.5, tiempo.shape)

transformada = scipy.fftpack.rfft(señal) #Se calcula la transformada rápida de Fourier usando la función de SciPy.

transformadaM = np.abs(transformada) #Se calcula la magnitud de la transformada de Fourier de la señal de entrada.

#Se calculan las frecuencias respectivas de la señal de entrada con la frecuencia de muestreo
frec_señal = np.linspace(0.0, 1.0/(2.0*Tm), señal.size)

#Se define la función para filtrar la señal y obtener la señal filtrada
def Filtrar_Señal(señal, umbral):
    #Filtro pasa bajas usando su respuesta en frecuencia.
    def pasaBajas(x, w0):
        return w0/(w0 + 1j*x)
    
    #Se genera una copia de la transformada obtenida para filtrar
    señalFiltrada = señal.copy()
    
    #Se filtran la señal usando la fft en cada una de las frecuencias
    for i in range(0, señalFiltrada.shape[0]):
        señalFiltrada[i] *= pasaBajas(frec_señal[i], umbral)

    #Se retorna el resultado del filtrado
    return señalFiltrada

#Se obtiene la fft de la señal filtrada.
#Se filtra con un umbral (frecuencia de corte) de 550 Hz, ya que las señales deseadas tienen una frecuencia menor a esta y no van a ser tan atenuadas por este filtro pasa bajas.
transformadaFiltrada = Filtrar_Señal(transformada, 550)

#Se calcula la magnitud de la transformada de Fourier de la señal filtrada.
transformadaFiltM = np.abs(transformadaFiltrada)

#Se calcula la señal filtrada de la transformada de Fourier filtrada.
señalFiltrada = scipy.fftpack.irfft(transformadaFiltrada)

#Se grafica la señal de entrada.
fig1, ax1 = plt.subplots(1, 1)
ax1.plot(tiempo,señal)
ax1.set_title('Señal de entrada')

#Se grafica la transformada de Fourier de la señal de entrada.
fig2, ax2 = plt.subplots(1, 1)
ax2.plot(frec_señal, (2.0/señal.size)*transformadaM)
ax2.set_title('Transformada de Fourier de la señal de entrada (Magnitud)')

#Se grafica la señal filtrada.
fig3, ax3 = plt.subplots(1, 1)
ax3.plot(tiempo,señalFiltrada)
ax3.set_title('Señal filtrada')

#Se grafica la transformada de Fourier de la señal filtrada.
fig3, ax3 = plt.subplots(1, 1)
ax3.plot(frec_señal, (2.0/señal.size)*transformadaFiltM)
ax3.set_title('Transformada de Fourier de la señal filtrada (Magnitud)')

plt.show()