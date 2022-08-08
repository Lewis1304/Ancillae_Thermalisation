from qiskit import *
import matplotlib.pyplot as plt
import numpy as np
from qiskit.quantum_info.operators import Operator
from qiskit.tools.monitor import job_monitor
import csv
from numpy import linalg as LA
import matplotlib.ticker as mticker

def post_processing_angle():
    data = np.genfromtxt('data_sim_TESTING.csv', delimiter="\t",skip_header=1)
    data_physical = np.genfromtxt('data_physical2.csv',delimiter="\t",skip_header=1)
    x = np.linspace(0,np.pi/2,25)
    x_physical = np.linspace(0,np.pi/2,5)

    for i,k in zip([0,1,2,5],['.','^','x','*']):
        i_rounds = [data[j] for j in range(i,len(data),8)]
        arctan_xz = [np.arctan2(j[2],j[4])/2 for j in i_rounds[:-2]] #factor of two since every angle is multiplied by 2 on the bloch sphere


        if i == 0 :#these conditions just shift the result to another solution which fits the curve
            arctan_xz += [(np.arctan(j[2]/j[4]) + np.pi)/2 for j in i_rounds[-2:]]
            plt.plot(x,arctan_xz,label = '${} \ round$'.format(i+1),linewidth = 0,marker = k,color = 'tab:grey')
        elif i == 2:
            arctan_xz += [(np.arctan(j[2]/j[4]) + np.pi)/2 for j in i_rounds[-2:]]
            plt.plot(x,arctan_xz,label = '${} \ rounds$'.format(i+1),linewidth = 0,marker = k,color = 'tab:grey')
        elif i == 5:
            arctan_xz += [np.arctan2(j[2],j[4])/2 for j in i_rounds[-2:-1]] #factor of two since every angle is multiplied by 2 on the bloch sphere
            arctan_xz += [(np.arctan2(j[2],j[4]) + np.pi)/2 for j in i_rounds[-1:]] #factor of two since every angle is multiplied by 2 on the bloch sphere

        else:
            arctan_xz += [np.arctan2(j[2],j[4])/2 for j in i_rounds[-2:]]
            plt.plot(x,arctan_xz,label = '${} \ rounds$'.format(i+1),linewidth = 0,marker = k,color = 'tab:grey')

    """
    for i,k in zip(range(3),['o','s','D']):
        i_rounds = [data_physical[j] for j in range(i,len(data_physical),3)]
        arctan_xz = [np.arctan2(j[2],j[4])/2 for j in i_rounds]
        if i == 0:
            plt.plot(x_physical,arctan_xz,linewidth = 0, marker = k,markersize = 8,fillstyle = 'none',label = '${} \ round \ QC$'.format(i+1),color = 'k')
        else:
            plt.plot(x_physical,arctan_xz,linewidth = 0, marker = k,markersize = 8,fillstyle = 'none',label = '${} \ rounds \ QC$'.format(i+1),color = 'k')
    """

    plt.ylabel('$arctan(\\frac{\\langle x \\rangle}{\\langle z \\rangle})$',labelpad=-4)
    plt.xlabel('$\\theta \ (radians)$',labelpad=-2)

    positions = (0,np.pi/8,np.pi/4, 3*np.pi/8,np.pi/2)
    labels = ("$0$", "$\\frac{\\pi}{8}$", "$\\frac{\\pi}{4}$","$\\frac{3\\pi}{8}$","$\\frac{\\pi}{2}$")
    plt.xticks(positions, labels,fontsize = 12)

    plt.plot(x,[q(theta) for theta in x],label = '$arctan(tan^2(\\theta))$',color = 'k')
    plt.legend()



    """
    x = range(1,4)
    data = np.genfromtxt('data_physical2.csv',delimiter="\t",skip_header=1)
    a = plt.axes([0.65, 0.25, .2, .2])
    for k,j in enumerate([4]):
        results = []
        theta = np.linspace(0,np.pi/2,5)[2] #theta = pi/4
        for i in x:
            X_exp = data[j + (i-1)][2]
            Y_exp = data[j + (i-1)][3]
            Z_exp = data[j + (i-1)][4]

            comp = computed_fidelity(theta,X_exp,Y_exp,Z_exp)
            results.append(comp)
        plt.plot(x,results,color = 'k')
    """
    plt.xticks(fontsize=8)
    plt.title('Fidelity',fontsize = 10)
    plt.yticks(fontsize=8)
    plt.xlabel('Number of Rounds',fontsize = 8)
    #plt.ylabel('Fidelity',fontsize = 8)

    #plt.xlabel("$Number of Rounds$")
    #plt.ylabel("Trace Distance")
    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))

    plt.text(2.3,0.65,'$\\theta = \\frac{\\pi}{8}$',fontsize = 10, bbox=dict(facecolor='none', edgecolor='black'))
    plt.savefig('Perceptron_angle_testing.pdf')
    plt.show()


def q(theta):
    return np.arctan(np.tan(theta)**2)

def p(theta):
    return np.cos(theta)**4+ np.sin(theta)**4

def analytic_evs(theta, N):
    """analytic expectation values
    """
    #return (1-(1-p(theta))**N)*np.array([2*np.cos(theta)**2*np.sin(theta)**2, 0, np.cos(theta)**4-np.sin(theta)**4])/p(theta)+(1-p(theta))**N*np.array([0, 0, 1])
    return (1-(1-p(theta))**N) * np.array([2 * np.cos(q(theta)) * np.sin(q(theta)),0, np.cos(q(theta))**2 - np.sin(q(theta))**2]) + (1-p(theta))**N * np.array([0,0,1])


def testing_expectation_values():
    x = np.linspace(0,np.pi/2,25)
    for r in range(1,4):
        results = []
        for i in x:
            X_exp = experiment_QC(i,r,'x')
            Y_exp = experiment_QC(i,r,'y')
            Z_exp = experiment_QC(i,r,'z')
            results.append([X_exp, Y_exp, Z_exp])
        plt.plot(x,np.array(results),label = 'Round: {}'.format(r))
        plt.plot(x, np.array([analytic_evs(i, r) for i in x]), linestyle='dashed', label='analytic')
        print('{} / 3'.format(r))
    plt.show()

def computed_fidelity(theta,X_exp,Y_exp,Z_exp):
    computed_list = np.array([X_exp,Y_exp,Z_exp])
    expected_list = np.array([2*np.cos(theta)**2*np.sin(theta)**2, 0, np.cos(theta)**4-np.sin(theta)**4])/p(theta)
    return 0.5*(1+np.dot(computed_list, expected_list))

def analytic_fidelity(theta,n_rounds):
    return (1 - (1-p(theta))**n_rounds*(1-np.cos(theta)**4/p(theta)))

def post_processing_fidelity():
    results = []
    x = range(1,9)
    data = np.genfromtxt('data_sim.csv', delimiter="\t",skip_header=1)
    theta = np.pi/4 #Doing this for pi/4 (worst case) >>> starts @ 8*12'th row in data

    for i in x:
        X_exp = data[8*12 + (i - 1)][2]
        Y_exp = data[8*12 + (i - 1)][3]
        Z_exp = data[8*12 + (i - 1)][4]

        comp = computed_fidelity(theta,X_exp,Y_exp,Z_exp)
        results.append(comp)

    plt.plot(x,results,label = "$\\frac{1}{2}(1 + \\langle \\psi |R\\psi \\rangle)$")
    plt.plot(x,[analytic_fidelity(theta,n) for n in x], linestyle='dashed', label = '$Analytic, \ eq. \ x$')
    plt.xlabel("$Number\ of\ Ancillae$")
    plt.ylabel("$Fidelity$")
    plt.legend()
    plt.show()


post_processing_angle()

