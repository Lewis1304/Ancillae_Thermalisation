import matplotlib.pyplot as plt
from qiskit import *
import numpy as np
from qiskit.quantum_info.operators import Operator
import csv
from qiskit.circuit.library import MCMT
from qiskit.circuit.library import OR
from tqdm import tqdm
import matplotlib.ticker as mticker

def minus_iY():

    minus_iY = Operator(np.array([[0,-1],[1,0]]))
    target = QuantumRegister(1,'t_qbit')
    qc = QuantumCircuit(target)
    qc.unitary(minus_iY,[*target])

    return qc

def U(theta):

    """
    Post-select unitary U
    """

    ancillae = QuantumRegister(1,'a_qbit')
    target = QuantumRegister(1,'t_qbit')

    qc = QuantumCircuit(ancillae,target)

    qc.ry(2*theta,[*ancillae])
    qc.append(MCMT(minus_iY(),1,1),[*ancillae,*target]) #qc.u3(np.pi,0,0) = -iY
    qc.ry(-2*theta,[*ancillae])

    return qc.to_instruction()

def controlled_U(theta):
    """
    Controlled post-select unitary U
    """
    new_qubit = QuantumRegister(1,'n_qbit')
    old_qubit = QuantumRegister(1,'o_qbit')
    target = QuantumRegister(1,'t_qbit')

    qc = QuantumCircuit(old_qubit,new_qubit,target)

    qc.ry(2*theta,new_qubit)
    qc.append(MCMT(minus_iY(),2,1),[old_qubit,new_qubit,target])
    qc.ry(-2*theta,new_qubit)

    return qc.to_instruction()

def experiment(theta,n,measure=None):
    register_qubits = QuantumRegister(n+1)
    target_qubit = QuantumRegister(1)
    c = ClassicalRegister(1)
    circ = QuantumCircuit(register_qubits,target_qubit,c)

    #Initial Round
    circ.append(U(theta),[register_qubits[0],target_qubit]) #qc.u3(np.pi,0,0) = -iY

    #Subsequent Rounds
    for i in range(1,n+1):
        circ.cry(np.pi/2,register_qubits[i-1],*target_qubit)
        circ.append(controlled_U(theta),[register_qubits[i-1],register_qubits[i],target_qubit])

    #Pauli Measurements
    if measure == 'x':
        circ.h(target_qubit)
    elif measure == 'y':
        circ.sdg(target_qubit)
        circ.h(target_qubit)
    circ.measure(target_qubit,c)

    shots = 8192
    qcomp = Aer.get_backend('qasm_simulator')
    job = execute(circ, backend = qcomp, shots = shots)
    result = job.result()
    result_dictionary = result.get_counts(circ)
    probs = {}
    for output in ['0','1']:
        if output in result_dictionary:
            probs[output] = result_dictionary[output]
        else:
            probs[output] = 0
    return (probs['0'] -  probs['1']) / shots

def acquiring_data():

    with open('results.csv', 'w') as csvfile:

        writer = csv.writer(csvfile, delimiter="\t")
        writer.writerow(['Theta', 'rounds', '<X>', '<Y>', '<Z>'])

        for theta in tqdm(np.linspace(0,np.pi/2,num = 25)): #noise
            for n in [1,2,3,6]:
                X_exp = experiment(theta,n,measure='x')
                Y_exp = experiment(theta,n,measure='y')
                Z_exp = experiment(theta,n,measure='z')
                writer.writerow([theta,n,X_exp,Y_exp,Z_exp])

def post_processing_angle():
    #Note: Some results (at the end) are shifted by pi/2 i.e. another solution
    data = np.genfromtxt('results.csv', delimiter="\t",skip_header=1)
    x = np.linspace(0,np.pi/2,25)

    for i,k,m in zip(range(4),['.','^','x','*'],[1,2,3,6]):
        i_rounds = [data[j] for j in range(i,len(data),4)]
        arctan_xz = [np.arctan2(j[2],j[4])/2 for j in i_rounds] #factor of two since every angle is multiplied by 2 on the bloch sphere
        plt.plot(x,arctan_xz,label = '${} \ round$'.format(m),linewidth = 0,marker = k,color = 'tab:grey')
    plt.plot(x,[np.arctan(np.tan(i)**2) for i in x],label = '$arctan(tan^2(\\theta))$',color = 'k')

    plt.ylabel('$arctan(\\frac{\\langle x \\rangle}{\\langle z \\rangle})$',labelpad=-4)
    plt.xlabel('$\\theta \ (radians)$',labelpad=-2)

    positions = (0,np.pi/8,np.pi/4, 3*np.pi/8,np.pi/2)
    labels = ("$0$", "$\\frac{\\pi}{8}$", "$\\frac{\\pi}{4}$","$\\frac{3\\pi}{8}$","$\\frac{\\pi}{2}$")
    plt.xticks(positions, labels,fontsize = 12)
    plt.legend()

    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)

    plt.show()

if __name__ == '__main__':
    acquiring_data()
    post_processing_angle()
