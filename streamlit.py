import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from scipy.fft import fft
import folium

# Lataa data
location_data = pd.read_csv('Location.csv')
df_step = pd.read_csv('Linear Acceleration.csv')

#Suodatetaan datasta selvästi kävelytaajuutta suurempitaajuuksiset vaihtelut pois
from scipy.signal import butter,filtfilt
def butter_lowpass_filter(data, cutoff, nyq, order):
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

#Filttereiden parametrit:
T = df_step['Time (s)'][len(df_step['Time (s)'])-1] - df_step['Time (s)'][0] #Koko datan pituus
n = len(df_step['Time (s)']) #Datapisteiden lukumäärä
fs = n/T #Näytteenottotaajuus (olettaen jotakuinkin vakioksi)
nyq = fs/2 #Nyqvistin taajuus
order = 3 #Kertaluku
cutoff = 1/(0.02) #Cutt-off taajuus

filtered_signal =  butter_lowpass_filter(df_step['Linear Acceleration y (m/s^2)'], cutoff, nyq, order)


#Lasketaan jaksojen määrä signaalissa (ja sitä kautta askelten määrä) laskemalla signaalin nollakohtien ylitysten määrä. 
#Nolla ylitetään kaksi kertaa jokaisen jakson aikana
jaksot = 0
for i in range(len(filtered_signal)-1):
    if filtered_signal[i]/filtered_signal[i+1] < 0:
        jaksot = jaksot + 1
steps = np.floor(jaksot/2)

# Laske Fourier-analyysi
N = len(df_step['Linear Acceleration y (m/s^2)'])  # Datapisteiden lukumäärä signaalissa
dt = 1 / fs  # Näytteenottoväli, datapisteiden ajallinen välimatka

# Lasketaan Fourier
fourier = np.fft.fft(df_step['Linear Acceleration y (m/s^2)'], N)

# Lasketaan tehospektri (Power Spectral Density, PSD)
psd = np.abs(fourier) ** 2 / N  # Lasketaan teho oikein

# Taajuudet
freq = np.fft.fftfreq(N, dt)

# Lasketaan askelmäärä tehospektrin huippujen perusteella
peaks = (psd > np.mean(psd))  # Valitaan taajuudet, jotka ylittävät keskiarvon
step_count_fft = np.sum(peaks[:N // 2])  # Lasketaan vain positiiviset taajuudet

# Kuljetun matkan laskenta
# Oulun leveysaste radiaaneina
latitude_rad = np.radians(65.01)

# Kuljetun matkan laskenta Oulussa
location_data['Latitude Distance'] = location_data['Latitude (°)'].diff() * 111320  # Leveysasteen muutos metreinä
location_data['Longitude Distance'] = location_data['Longitude (°)'].diff() * (111320 * np.cos(latitude_rad))  # Pituusasteen muutos metreinä

# Yhdistetään matkat
location_data['Distance'] = np.sqrt(
    location_data['Latitude Distance'] ** 2 +
    location_data['Longitude Distance'] ** 2
)

# Kokonaismatka
total_distance = location_data['Distance'].sum()

total_distance = location_data['Distance'].sum()

# Laske keskinopeus
time_taken = location_data['Time (s)'].max() - location_data['Time (s)'].min()
average_speed = total_distance / time_taken if time_taken > 0 else 0  # m/s
average_speed_kmh = average_speed * 3.6  # m/s to km/h

# Laske askelpituus
step_length = total_distance / steps if steps > 0 else 0  # Estä nollalla jakaminen

# Visualisointi
# Luo kartta
m = folium.Map(location=[location_data['Latitude (°)'].mean(), location_data['Longitude (°)'].mean()], zoom_start=15)
points = list(zip(location_data['Latitude (°)'], location_data['Longitude (°)']))
folium.PolyLine(points, color='blue', weight=2.5, opacity=1).add_to(m)
m.save('map.html')

# Streamlit
st.title("Kävelyliikemittausanalyysi")
st.write(f"Askelmäärä suodatetusta datasta: {steps}")
st.write(f"Askelmäärä Fourier-analyysin perusteella: {step_count_fft}")
st.write(f"Keskinopeus: {average_speed:.2f} m/s")
st.write(f"Keskinopeus: {average_speed_kmh:.2f} km/h")
st.write(f"Kuljettu matka: {total_distance:.2f} m")
st.write(f"Askelpituus: {step_length:.2f} m")

st.subheader("Suodatettu Kiihtyvyysdata")
st.line_chart(df_step[['Time (s)', 'Linear Acceleration y (m/s^2)']].set_index('Time (s)'))

st.subheader("Tehospektritiheys")
st.line_chart(pd.DataFrame({'Frequency (Hz)': freq[:N // 2], 'PSD': psd[:N // 2]}).set_index('Frequency (Hz)'))

st.subheader("Reitti kartalla")
st.components.v1.html(open('map.html', 'r').read(), height=500)
