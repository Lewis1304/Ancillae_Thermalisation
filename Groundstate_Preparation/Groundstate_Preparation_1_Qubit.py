from qiskit import *
import matplotlib.pyplot as plt
import numpy as np
from qiskit.quantum_info.operators import Operator
from qiskit.providers.aer import noise
import csv
import matplotlib.ticker as mticker
from qiskit.providers.aer.noise.errors import thermal_relaxation_error
from qiskit.providers.aer.noise import NoiseModel
from qiskit.circuit.library import MCMT
from qiskit.circuit.library import OR
from tqdm import tqdm

def Single_Controlled_Unitary(power):
    phases = [0,0.75]
    U = np.kron( np.diag([0,1]),np.diag([np.exp(2 * np.pi * 1.0j * k) for k in phases]) )
    U += np.diag([1,1,0,0])
    return Operator(U**(2**power))

def Unitary(power):
    phases = [0,0.75]
    U = np.diag([np.exp(2 * np.pi * 1.0j * k) for k in phases])
    U = Operator(U**(2**power))
    target = QuantumRegister(1,'t_qbit')
    qc = QuantumCircuit(target)
    qc.unitary(U,[*target])
    return qc

def Single_Controlled_V(i):
    V =  np.array([[1,0],[0,np.exp(0.75j * np.pi)]])**(2**i)
    c_v_array = np.kron(np.diag([0,1]),V) + np.diag([1,1,0,0])
    return Operator(c_v_array)

def Single_Controlled_V_dagger(i):
    V =  np.array([[1,0],[0,np.exp(0.75j * np.pi)]])**(2**i)
    c_v_array_dagger = np.kron(np.diag([0,1]),V.conj().T) + np.diag([1,1,0,0])
    return Operator(c_v_array_dagger)

def qft_dagger(circ, n):
    """n-qubit QFTdagger the first n qubits in circ"""
    # Don't forget the Swaps!
    for qubit in range(int(n/2)):
        circ.swap(qubit, n-qubit-1)
    for j in range(n,0,-1):
        k = n - j
        for m in range(k):
            circ.cu1(-math.pi/float(2**(k-m)), n-m-1, n-k-1)
        circ.h(n-k-1)

def experiment(p,t,measure_type=None,noise_model=None):

    n_register_qubits =  t + p*t
    n_state_qubits = 1
    n_or_qubits = (p+1) * (t-1) #additional OR gates for scrambling at the end

    register_qubits = QuantumRegister(n_register_qubits,'r_qbit')
    state_qubits = QuantumRegister(n_state_qubits,'s_qbit')
    or_qubits = QuantumRegister(n_or_qubits,'o_qbit')
    c = ClassicalRegister(1)

    circ = QuantumCircuit(register_qubits,or_qubits,state_qubits,c)

    circ.h(state_qubits) #initial input state

    circ.h(register_qubits)

    #Initial Round

    #Controlled Unitary
    for i,q in enumerate(register_qubits[:t][::-1]):
        circ.append(MCMT(Unitary(i),1,1),[q,state_qubits])

    #QFT-dagger
    for qubit in range(int(t/2)):
        circ.swap(qubit, t-qubit-1)
    for j in range(t,0,-1):
        k = t - j
        for m in range(k):
            circ.cu1(-np.pi/float(2**(k-m)), t-m-1, t-k-1)
        circ.h(t-k-1)

    #OR-Gate
    circ.x([*register_qubits[:t],*or_qubits[:t-1]])
    circ.ccx(register_qubits[0],register_qubits[1],or_qubits[0])
    for i in range(t-2):
        circ.x(or_qubits[i])
        circ.ccx(or_qubits[i],register_qubits[i+2],or_qubits[i+1])

    #Reset / Scrambling gate
    circ.ch(or_qubits[t-2],state_qubits)

    #Subsequent Rounds
    for j in range(1,p+1):

        #CCU
        for i,q in enumerate(register_qubits[t*j:t*(j+1)][::-1]):
            #circ.append(MCMT(Unitary(i),2,1),[or_qubits[j-1],q,state_qubits])
            circ.x(q)
            circ.cx(or_qubits[j-1],q)
            circ.append(MCMT(Unitary(i),1,1),[q,state_qubits])

        #QFT-dagger
        for qubit in range(int(t/2)):
            circ.swap(qubit + t*j, t-qubit-1 + t*j)
        for f in range(t,0,-1):
            k = t - f
            for m in range(k):
                circ.cu1(-np.pi/float(2**(k-m)), t-m-1 + t*j, t-k-1 + t*j)
            circ.h(t-k-1 + t*j)

        #Or Gate
        circ.x([*register_qubits[t*j:t*(j+1)],*or_qubits[(t-1)*j:(t-1)*(j+1)]])
        circ.ccx(register_qubits[t*j],register_qubits[t*j + 1],or_qubits[(t-1)*j])

        for i in range(t-2):
            circ.x(or_qubits[(t-1)*j + i])
            circ.ccx(or_qubits[(t-1)*j + i],register_qubits[t*j + 2 + i],or_qubits[(t-1)*j + i + 1])

        #Scrambling gate
        circ.ch(or_qubits[(t-1)*j + t-2],state_qubits)

    #Pauli Measurements
    if measure_type == 'x':
        circ.h(state_qubits)
    elif measure_type == 'y':
        circ.sdg(state_qubits)
        circ.h(state_qubits)

    circ.measure(state_qubits,c)
    #IBMQ.load_account()
    #provider = IBMQ.get_provider('ibm-q')
    #qcomp = provider.get_backend('ibmq_qasm_simulator')
    qcomp = Aer.get_backend('qasm_simulator')

    shots = 8192

    if noise_model is None:
        job = execute(circ, backend = qcomp,shots = shots)
    else:
        job = execute(circ, backend = qcomp, shots = shots,noise_model=noise_model,basis_gates=noise_model.basis_gates)

    #print(job_monitor(job))
    result = job.result()
    result_dictionary = result.get_counts(circ)

    probs = {}
    for output in ['0','1']:
        if output in result_dictionary:
            probs[output] = result_dictionary[output]
        else:
            probs[output] = 0

    return (probs['0'] -  probs['1']) / shots

def thermal_noise(n_qubits,mu1,mu2,sig):
    num = n_qubits
    # T1 and T2 values for qubits 0-3
    T1s = np.random.normal(mu1, sig, num) # Sampled from normal distribution mean 50 microsec
    T2s = np.random.normal(mu2, sig, num)  # Sampled from normal distribution mean 50 microsec

    # Truncate random T2s <= T1s
    T2s = np.array([min(T2s[j], 2 * T1s[j]) for j in range(num)])

    # Instruction times (in nanoseconds)
    time_u1 = 0   # virtual gate
    time_u2 = 50  # (single X90 pulse)
    time_u3 = 100 # (two X90 pulses)
    time_cx = 300
    time_reset = 1000  # 1 microsecond
    time_measure = 1000 # 1 microsecond

    # QuantumError objects
    errors_reset = [thermal_relaxation_error(t1, t2, time_reset)
                    for t1, t2 in zip(T1s, T2s)]
    errors_measure = [thermal_relaxation_error(t1, t2, time_measure)
                      for t1, t2 in zip(T1s, T2s)]
    errors_u1  = [thermal_relaxation_error(t1, t2, time_u1)
                  for t1, t2 in zip(T1s, T2s)]
    errors_u2  = [thermal_relaxation_error(t1, t2, time_u2)
                  for t1, t2 in zip(T1s, T2s)]
    errors_u3  = [thermal_relaxation_error(t1, t2, time_u3)
                  for t1, t2 in zip(T1s, T2s)]
    errors_cx = [[thermal_relaxation_error(t1a, t2a, time_cx).expand(
                 thermal_relaxation_error(t1b, t2b, time_cx))
                  for t1a, t2a in zip(T1s, T2s)]
                   for t1b, t2b in zip(T1s, T2s)]

    # Add errors to noise model
    noise_thermal = NoiseModel()
    for j in range(num):
        noise_thermal.add_quantum_error(errors_reset[j], "reset", [j])
        noise_thermal.add_quantum_error(errors_measure[j], "measure", [j])
        noise_thermal.add_quantum_error(errors_u1[j], "u1", [j])
        noise_thermal.add_quantum_error(errors_u2[j], "u2", [j])
        noise_thermal.add_quantum_error(errors_u3[j], "u3", [j])
        for k in range(num):
            noise_thermal.add_quantum_error(errors_cx[j][k], "cx", [j, k])

    return noise_thermal

def computed_fidelity(X_exp,Y_exp,Z_exp):
    computed_list = np.array([X_exp,Y_exp,Z_exp])
    expected_list = np.array([0,0,1])
    return np.real(0.5*(1+np.dot(computed_list, expected_list)))

def results(max_rounds):

    t = 2 #two qubits per register
    n = 1 #one qubit Hamiltonian
    x = range(0,max_rounds+1)

    with open('Groundstate_Preparation_Single_Qubit_Hamiltonian.csv', 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter="\t")
        writer.writerow(['Rounds', 'mu_1','mu2', 'Fidelity with noise'])

        for p in tqdm(x):
            for mu1,mu2,sigma in zip([50e3,180e3,180e4],[70e3,200e3,200e4],[10e3,36e3,10e4]):
                n_qubits = n + (2*t - 1)*p
                noise_thermal = thermal_noise(n_qubits,mu1,mu2,sigma)

                X_exp = experiment(p,t,'x',noise_thermal)
                Y_exp = experiment(p,t,'y',noise_thermal)
                Z_exp = experiment(p,t,'z',noise_thermal)
                results_noise = computed_fidelity(X_exp,Y_exp,Z_exp)

                writer.writerow([p,mu1,mu2,results_noise])

def plot_results(max_rounds):
    x = range(0,max_rounds+1)
    data = np.genfromtxt('Groundstate_Preparation_Single_Qubit_Hamiltonian.csv',delimiter="\t",skip_header=1)

    for i,k in zip(range(3),['High','Med','Low']):
        results_noise = [data[j][3] for j in range(i,len(data),3)]
        plt.plot(x,results_noise,label = '{}'.format(k))

    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
    plt.xlabel('$Number \ of \ Rounds$')
    plt.ylabel('$Fidelity$')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    max_rounds = 2
    results(max_rounds)
    plot_results(max_rounds)
