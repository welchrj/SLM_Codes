# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 16:26:24 2019

@author: welch
"""
import matplotlib.pyplot as plt
import numpy
import os


def file_read(file):
    data = []
    with open(file,'r') as f:
        for line in f:
            if '[Data]' in line:
                for line in f:
                    data.append(line)
    return data

def split_table(data):
    data = [x.split(',') for x in data]
    x, y = [], []
    for i in range(1,len(data)):
        x.append(float(data[i][0]))
        y.append(float(data[i][1]))
    x = numpy.array(x)
    y= numpy.array(y)
    return x, y

def normalize(data):
    data[0:] = (data[0:]-min(data))/(max(data)-min(data))*100
# =============================================================================
#     data[0:] = (data[0:]/max(data))*100
# =============================================================================
    return data

def plot(x1, y1, x2, y2, pdf_theta, pdf_intensity, pdf2, pdfi2, name='figure.svg'):
    plt.figure()
    plt.plot(x2, y2,'r',linewidth=0.5, label = 'SLM')
    plt.plot(x1, y1,'k',linewidth=0.5, label = 'Meltgrown')
    plt.title('XRD')
    plt.xlabel('2theta')
    plt.ylabel('Intensity')
    plt.ylim(0,100)
    plt.xlim(20,70)
    plt.legend()

    for i in range(len(pdf_theta)):
        plt.axvline(x = pdf_theta[i], ymin=0, ymax=pdf_intensity[i]/100,
                linewidth='0.5',color='blue')
    for i in range(len(pdf2)):
        plt.axvline(x = pdf2[i], ymin=0, ymax=pdfi2[i]/100,
                linewidth='0.5',color='orange')
    print(os.getcwd())
    print('up')
    plt.savefig(name, format = 'svg')


def xrd_pdf(theta_min, theta_max):
    os.chdir('C:\\Users\\welch\\Documents\\GWU\Research\\Materials_Characterization\\USF_2N\\XRD\\PDF')
    pdf_doped = 'Bi2Se0.3Te2.7_XRD_PDF.csv'
    pdf_undoped = 'Bi2Te3_XRD_PDF.csv'
    theta_1, intensity1 = numpy.loadtxt(pdf_doped, delimiter=',', skiprows = 4,
                        usecols = (0,1), unpack = True)
    theta_2, intensity2 = numpy.loadtxt(pdf_undoped, delimiter=',', skiprows = 4,
                        usecols = (0,1), unpack = True)

    theta_1, intensity1 = pdf_range(theta_1, intensity1, theta_min, theta_max)
    theta_2, intensity2 = pdf_range(theta_2, intensity2, theta_min, theta_max)

    
    return theta_1, intensity1, theta_2, intensity2

def pdf_range(pdf_theta, pdf_intensity, theta1, theta2):
    mask = numpy.where((pdf_theta >= theta1) & (pdf_theta <= 70))
    pdf_theta = pdf_theta[mask]
    pdf_intensity = pdf_intensity[mask]

    return pdf_theta, pdf_intensity

# =============================================================================
#      = np.loadtxt(file1,delimiter=',',unpack=True)
# =============================================================================

def main():
    os.chdir('C:\\Users\\welch\\Documents\\GWU\Research\\Materials_Characterization\\USF_2N\\XRD\\XRD_7_19_2019\\')
    a = os.getcwd() + '\\'
    file1 = a + 'USF-2N-HP-51B_07_19_19.txt'
    file2 = a + 'USF-2N-SLM-51B_07_19_19.txt'
    print(file1)
    print(file2)
    angle_hp, count_hp = split_table(file_read(file1))
    angle_slm, count_slm = split_table(file_read(file2))
    count_hp = normalize(count_hp)
    count_slm = normalize(count_slm)
    theta_min = min(angle_hp)
    theta_max = max(angle_hp)

    pdfN_theta, pdfN_i, pdfN_theta2, pdfN_i2 = xrd_pdf(theta_min, theta_max)
    #pdf_theta, pdf_i = xrd_pdf(theta_min, theta_max)
    plot(angle_hp, count_hp, angle_slm, count_slm,
         pdfN_theta, pdfN_i, pdfN_theta2, pdfN_i2, name='XRD_20190719')


# =============================================================================
#     os.chdir('C:\\Users\\welch\\Documents\\GWU\Research\\Materials_Characterization\\USF_2N\\XRD\\XRD_8_15_2019\\')
#     a = os.getcwd()
#     file1 = a + '\\USF-2N-HP-51B_0.01inc_60-70deg_08_15_19.txt'
#     file2 = a + '\\USF-2N-SLM-51B_0.01inc_60-70deg_08_15_19.txt'
#     print(file1)
#     print(file2)
#     data_hp2 = file_read(file1)
#     data_slm2 = file_read(file2)
#     angle_hp2, count_hp2 = split_table(data_hp2)
#     angle_slm2, count_slm2 = split_table(data_slm2)
#     count_hp2 = normalize(count_hp2)
#     count_slm2 = normalize(count_slm2)
#     theta_min = min(angle_hp2)
#     theta_max = max(angle_hp2)
#     pdf_theta2, pdf_i2 = xrd_pdf(theta_min, theta_max)
#     pdf_i2 = normalize(pdf_i2)
#     plot(angle_hp2, count_hp2, angle_slm2, count_slm2,
#          pdf_theta2, pdf_i2, name='XRD_60-70deg_20190815')
# =============================================================================



if __name__ == '__main__':
    main()
# =============================================================================
# plt.figure(1)
# plt.plot(angle_hp,count_hp,'k',linewidth=0.5, label = 'Meltgrown')
# plt.plot(angle_slm,count_slm,'r',linewidth=0.5, label = 'SLM')
# plt.title('XRD')
# plt.xlabel('2theta')
# plt.ylabel('Intensity')
# plt.legend()
# plt.savefig('XRD.png',dpi=1047)
# =============================================================================

# =============================================================================
# for i in range(len(data)):
#     data[i] = data[i].split(',')
# =============================================================================



