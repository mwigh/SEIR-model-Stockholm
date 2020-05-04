import os
import datetime
import pandas as pd

CSV_CASE_PATH = os.path.join('Data', 'Data_2020-04-10Ny.txt')
DATE_COLUMN = 'Datum'
INCIDENSE_COLUMN = 'Incidens'
MEAN_LENGTH_SICK = 5
MEAN_LENGTH_INCUBATION=5.1

N=2374550#report page 9#int(648557/0.27+616655/0.26)/2 #Mean from Tabell 1 page 13 #approx 2.386M
i0=1
START_DATE = datetime.datetime(2020, 2, 17)
END_DATE_OPTIMIZE = datetime.datetime(2020, 4, 10)
END_DATE_SIMULATION = datetime.datetime(2020, 8, 10)
MIDDLE_THETA = datetime.datetime(2020, 3, 16)

p0 = 1-0.987 #Calibrated somehow to fit 2.5% in end march 
q0=0.11
delta=0.16
epsilon=-0.19
theta=10.9
rho = 1/MEAN_LENGTH_INCUBATION
gamma = 1/MEAN_LENGTH_SICK
t_b=(MIDDLE_THETA-START_DATE).days
print(t_b)
