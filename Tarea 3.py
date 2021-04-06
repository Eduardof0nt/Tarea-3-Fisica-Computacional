import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack

tiempoMuestreo = 0.0001 #Tiempo de muestreo
umbral = 0.5 #Umbral de amplitud de 0.5

tiempo = np.arange(0.0,0.05,tiempoMuestreo) #Se crea un arreglo que contiene todos los pasos discretos de tiempo de la señal que se va a generar.

#Se crea la señal que consiste de una onda senoidal con frecuencia de 440 Hz (un La en la cuarta octava del piano) y amplitud de 5 y otra con una frecuencia de 523.25 Hz (un Do en la quinta octava del piano) y amplitud de 3.
#También se le añade ruido usando la función random.normal de NumPy, este con una media de 0, una desviación estandar de 0.5 y una ampitud de 2.
señal = 5*np.sin(2*np.pi*440*tiempo) + 3*np.sin(2*scipy.pi*523.25*tiempo) + 2*np.random.normal(0, 0.5, tiempo.shape)

transformada = scipy.fftpack.rfft(señal) #Se calcula la transformada rápida de Fourier usando la función de SciPy.

transformadaMagnitud = np.abs(transformada) #Se calcula la magnitud de la transformada de Fourier de la señal de entrada.

#Se calculan las frecuencias respectivas de la señal de entrada con la frecuencia de muestreo
frec_señal = np.linspace(0.0, 1.0/(2.0*tiempoMuestreo), señal.size)

#Se define la función para filtrar la señal y obtener la señal filtrada
def Filtrar_Señal(señal, umbral):    
    #Se genera una copia de la transformada obtenida para filtrar
    señalFiltrada = señal.copy()
    
    #Se filtran la señal usando la fft en cada una de las frecuencias
    #Si la amplitud en el punto es menor al umbral, se deja en 0.
    for i in range(0, señalFiltrada.shape[0]):
        if (2.0/señal.size)*np.abs(señalFiltrada[i]) <= umbral:
            señalFiltrada[i] = 0

    #Se retorna el resultado del filtrado.
    return señalFiltrada

#Se obtiene la fft de la señal filtrada.
#Se filtra con el umbral definido, ya que las señales deseadas tienen una amplitud mayor a esta y no van a ser filtradas.
transformadaFiltrada = Filtrar_Señal(transformada, umbral)

#Se calcula la magnitud de la transformada de Fourier de la señal filtrada.
transformadaFiltradaMagnitud = np.abs(transformadaFiltrada)

#Se calcula la señal filtrada de la transformada de Fourier filtrada.
señalFiltrada = scipy.fftpack.irfft(transformadaFiltrada)

#Se grafica la señal de entrada.
fig1, ax1 = plt.subplots(1, 1)
ax1.plot(tiempo,señal)
ax1.set_title('Señal de entrada')
ax1.set_xlabel('Tiempo (s)')
ax1.set_ylabel('Amplitud (u.l.)')

#Se grafica la transformada de Fourier de la señal de entrada.
fig2, ax2 = plt.subplots(1, 1)
ax2.plot(frec_señal, (2.0/señal.size)*transformadaMagnitud)
ax2.set_title('Transformada de Fourier de la señal de entrada (Magnitud)')
ax2.set_xlabel('Frecuencia (Hz)')
ax2.set_ylabel('Amplitud (u.l.)')

#Se grafica la señal filtrada.
fig3, ax3 = plt.subplots(1, 1)
ax3.plot(tiempo,señalFiltrada)
ax3.set_title('Señal filtrada')
ax3.set_xlabel('Tiempo (s)')
ax3.set_ylabel('Amplitud (u.l.)')

#Se grafica la transformada de Fourier de la señal filtrada.
fig4, ax4 = plt.subplots(1, 1)
ax4.plot(frec_señal, (2.0/señal.size)*transformadaFiltradaMagnitud)
ax4.set_title('Transformada de Fourier de la señal filtrada (Magnitud)')
ax4.set_xlabel('Frecuencia (Hz)')
ax4.set_ylabel('Amplitud (u.l.)')

#Se grafican ambas señales para compararlas.
fig5, ax5 = plt.subplots(1, 1)
ax5.plot(tiempo,señal, 'b', label="Señal original")
ax5.plot(tiempo,señalFiltrada, 'r', label="Señal filtrada")
ax5.set_title('Señal original y filtrada')
ax5.set_xlabel('Tiempo (s)')
ax5.set_ylabel('Amplitud (u.l.)')
ax5.legend(loc='upper right')

plt.show()
