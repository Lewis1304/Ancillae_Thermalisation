import itertools
import csv
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
import numpy as np
from qiskit import IBMQ, Aer
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, execute
from qiskit.quantum_info.operators import Operator
from numpy import linalg as LA
from qiskit.providers.aer.noise.errors import thermal_relaxation_error
from qiskit.providers.aer.noise import NoiseModel
from tqdm import tqdm

###############################################################################
#Creating evolution operator U(2) = exp(-iH)
def granschmit(X): #Creates O out of linearly independent vectors (A)
    Q, R = np.linalg.qr(X)
    return Q

def u2(): #creates U(2) via similarity transformation ODO^-1
    O = np.eye(4) + np.random.normal(0,0.5,(4,4))
    W = granschmit(O)
    U = W @ np.diag([1,1j,-1,-1j]) @ W.T.conj()
    return U
###############################################################################

def u2_gs(U): #groundstate of hamiltonian. i.e. eigenstate corresponding to eigenvalue 1
    w, v = LA.eig(U)
    return v[:,np.where(np.round(w) == 1)[0][0]]

def Single_Controlled_Unitary(U):
    Q = np.kron( np.diag([0,1]),U )
    Q += np.diag([1,1,1,1,0,0,0,0])
    return Operator(Q)

def Two_Controlled_Unitary(U):
    Q = np.kron( np.diag([0,0,0,1]),U)
    Q += np.diag([1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0])
    return Operator(Q)

def thermal_noise(mu1,mu2,sigma,p,t):
    num = 2 + (2*t - 1)*p

    # T1 and T2 values for qubits 0-num
    T1s = np.random.normal(mu1, sigma, num) # Sampled from normal distribution mean mu1 microsec, sigma = 10e3
    T2s = np.random.normal(mu2, sigma, num)  # Sampled from normal distribution mean mu2 microsec

    # Truncate random T2s <= T1s
    T2s = np.array([min(T2s[j], 2 * T1s[j]) for j in range(num)])

    # Instruction times (in nanoseconds)
    time_u1 = 0   # virtual gate
    time_u2 = 50  # (single X90 pulse)
    time_u3 = 100 # (two X90 pulses)
    time_cx = 300

    #Time to implement controlled/controlled-controlled evolution unitary
    #Done this way to save on computational cost of simulating noise.
    time_cu = 2 * (2 * time_u1 + 2*time_cx + 2*time_u3 ) #time is 2* cu3(1).
    time_ccu = 3 * (2 * time_u1 + 2*time_cx + 2*time_u3) + 2* time_cx #ccu(2) can be decomposed into 2 ccu(1) which can be decomposed to 3 cu3(1) and 2 cx gates

    # QuantumError objects
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
    errors_cu = [[[thermal_relaxation_error(t1a, t2a, time_cu).expand(
                    thermal_relaxation_error(t1b,t2b,time_cu).expand(
                    thermal_relaxation_error(t1c,t2c,time_cu)))
                for t1a, t2a in zip(T1s,T2s)]
                for t1b, t2b in zip(T1s,T2s)]
                for t1c, t2c in zip(T1s,T2s)]

    if num > 5: #i.e. more than 1 iteration of PEA

        errors_ccu = [[[[thermal_relaxation_error(t1a,t2a,time_ccu).expand(
                    thermal_relaxation_error(t1b,t2b,time_ccu).expand(
                    thermal_relaxation_error(t1c,t2c,time_ccu).expand(
                    thermal_relaxation_error(t1d,t2d,time_ccu))))
                for t1a,t2a in zip(T1s,T2s)]
                for t1b,t2b in zip(T1s,T2s)]
                for t1c,t2c in zip(T1s,T2s)]
                for t1d,t2d in zip(T1s,T2s)]

    # Add errors to noise model
    noise_thermal = NoiseModel()

    for j in range(num):
        noise_thermal.add_quantum_error(errors_u1[j], "u1", [j])
        noise_thermal.add_quantum_error(errors_u2[j], "u2", [j])
        noise_thermal.add_quantum_error(errors_u3[j], "u3", [j])
        for k in range(num):
            noise_thermal.add_quantum_error(errors_cx[j][k], "cx", [j, k])

    #Only Need CU and CCU gate noise between qubits that actually implement this gate!

    state_qubit_0 = num - 2
    state_qubit_1 = num - 1

    for q in range(t): #CU only acts on state qubits and register from first round (in this order)
        noise_thermal.add_quantum_error(errors_cu[state_qubit_1][state_qubit_0][q], "cu", [state_qubit_1,state_qubit_0,q])

    if num > 5:
        for j in range(1,p): #CCU acts on state qubits, or qubit from previous round and fresh register qubits (in this order)
            for q in range(t*j,t*(j+1)):
                noise_thermal.add_quantum_error(errors_ccu[state_qubit_1][state_qubit_0][p*t + (j-1)][q], "ccu", [state_qubit_1, state_qubit_0, p*t + (j-1), q])

    noise_thermal.add_basis_gates(['unitary'])
    return noise_thermal

def experiment(p,t,sigma_1,sigma_2,U,noise_model):
    #Note: p=1 is PEA + Scrambling gate (This is p=0 in other codes)
    n_register_qubits =  p*t
    n_state_qubits = 2
    n_or_qubits = p * (t-1)

    register_qubits = QuantumRegister(n_register_qubits)
    state_qubits = QuantumRegister(n_state_qubits)
    or_qubits = QuantumRegister(n_or_qubits)
    c = ClassicalRegister(2)

    circ = QuantumCircuit(register_qubits,or_qubits,state_qubits,c)

    circ.h(state_qubits) #initial input state


    #Initial Round
    circ.h(register_qubits)

    #Controlled U(2)
    for i,q in enumerate(register_qubits[:t][::-1]):
        for k in range(0,2**i):
            circ.unitary(Single_Controlled_Unitary(U),[state_qubits[1],state_qubits[0],q],label = 'cu')

    #QFT-dagger
    for qubit in range(int(t/2)):
        circ.swap(qubit, t-qubit-1)
    for j in range(t,0,-1):
        k = t - j
        for m in range(k):
            circ.cu1(-np.pi/float(2**(k-m)), t-m-1, t-k-1)
        circ.h(t-k-1)

    #Reset (OR + Scrambling gates)
    #OR-Gate
    circ.x(register_qubits[:t])
    circ.x(or_qubits[0])
    circ.ccx(register_qubits[0],register_qubits[1],or_qubits[0])
    #Scrambling gate
    circ.ch(or_qubits[0],state_qubits)

    #Subsequent Rounds
    for j in range(1,p):

        #CCU
        for i,q in enumerate(register_qubits[t*j:t*(j+1)][::-1]):
            for k in range(0,2**i):
                circ.unitary(Two_Controlled_Unitary(U),[state_qubits[1],state_qubits[0],or_qubits[j-1],q],label = 'ccu')

        #QFT-dagger
        for qubit in range(int(t/2)):
            circ.swap(qubit + t*j, t-qubit-1 + t*j)
        for f in range(t,0,-1):
            k = t - f
            for m in range(k):
                circ.cu1(-np.pi/float(2**(k-m)), t-m-1 + t*j, t-k-1 + t*j)
            circ.h(t-k-1 + t*j)

        #OR-Gate
        circ.x(register_qubits[t*j:t*(j+1)])
        circ.x(or_qubits[j])
        circ.ccx(register_qubits[t*j],register_qubits[t*j + 1],or_qubits[j])
        #Scrambling gate
        circ.ch(or_qubits[j],state_qubits)


    #Pauli Measurements (There's a neater way to do this!)
    if sigma_1 == 'z' and sigma_2 == 'I':
        circ.iden(state_qubits[0])
        circ.iden(state_qubits[1])

    if sigma_1 == 'x' and sigma_2 == 'I':
        circ.h(state_qubits[0])
        circ.iden(state_qubits[1])

    if sigma_1 == 'y' and sigma_2 == 'I':
        circ.sdg(state_qubits[0])
        circ.h(state_qubits[0])
        circ.iden(state_qubits[1])

    if sigma_1 == 'I' and sigma_2 == 'z':
        circ.swap(state_qubits[0],state_qubits[1])

    if sigma_1 == 'I' and sigma_2 == 'x':
        circ.swap(state_qubits[0],state_qubits[1])
        circ.h(state_qubits[0])
        circ.iden(state_qubits[1])

    if sigma_1 == 'I' and sigma_2 == 'y':
        circ.swap(state_qubits[0],state_qubits[1])
        circ.iden(state_qubits[1])
        circ.sdg(state_qubits[0])
        circ.h(state_qubits[0])

    if sigma_1 == 'z' and sigma_2 == 'z':
        circ.cx(state_qubits[1],state_qubits[0])

    if sigma_1 == 'z' and sigma_2 == 'x':
        circ.h(state_qubits[1])
        circ.iden(state_qubits[0])
        circ.cx(state_qubits[1],state_qubits[0])

    if sigma_1 == 'z' and sigma_2 == 'y':
        circ.sdg(state_qubits[1])
        circ.h(state_qubits[1])
        circ.iden(state_qubits[0])
        circ.cx(state_qubits[1],state_qubits[0])

    if sigma_1 == 'x' and sigma_2 == 'z':
        circ.iden(state_qubits[1])
        circ.h(state_qubits[0])
        circ.cx(state_qubits[1],state_qubits[0])

    if sigma_1 == 'x' and sigma_2 == 'x':
        circ.h(state_qubits[1])
        circ.h(state_qubits[0])
        circ.cx(state_qubits[1],state_qubits[0])

    if sigma_1 == 'x' and sigma_2 == 'y':
        circ.sdg(state_qubits[1])
        circ.h(state_qubits[1])
        circ.h(state_qubits[0])
        circ.cx(state_qubits[1],state_qubits[0])

    if sigma_1 == 'y' and sigma_2 == 'z':
        circ.iden(state_qubits[1])
        circ.sdg(state_qubits[0])
        circ.h(state_qubits[0])
        circ.cx(state_qubits[1],state_qubits[0])

    if sigma_1 == 'y' and sigma_2 == 'x':
        circ.h(state_qubits[1])
        circ.sdg(state_qubits[0])
        circ.h(state_qubits[0])
        circ.cx(state_qubits[1],state_qubits[0])

    if sigma_1 == 'y' and sigma_2 == 'y':
        circ.sdg(state_qubits[1])
        circ.h(state_qubits[1])
        circ.sdg(state_qubits[0])
        circ.h(state_qubits[0])
        circ.cx(state_qubits[1],state_qubits[0])

    circ.measure(state_qubits,c)

    shots = 8192
    #IBMQ.load_account()
    #provider = IBMQ.get_provider('ibm-q')
    #qcomp = provider.get_backend('ibmq_qasm_simulator')
    qcomp = Aer.get_backend('qasm_simulator')
    job = execute(circ, backend = qcomp, shots = shots,noise_model=noise_model,basis_gates=noise_model.basis_gates)
    #print(job_monitor(job))
    result = job.result()
    result_dictionary = result.get_counts(circ)
    probs = {}
    for output in ['00','01','10','11']:
        if output in result_dictionary:
            probs[output] = result_dictionary[output]
        else:
            probs[output] = 0

    return (probs['00'] + probs['11'] - probs['01'] - probs['10']) / shots

def groundstate_representation(U):
    I = np.eye(2)
    x = np.array([[0,1],[1,0]])
    y = np.array([[0,-1j],[1j,0]])
    z = np.array([[1,0],[0,-1]])
    gs = u2_gs(U)
    results = []
    a = []
    b = []
    T = []
    for j,k in itertools.product([I,z,x,y],[I,z,x,y]):

        experiment_result = gs @ np.kron(j,k) @ gs.conj().T
        if np.array_equal(j,I) and np.array_equal(k,I):
            continue

        if np.array_equal(j,I) and (not np.array_equal(k,I)):
            a.append(experiment_result)

        if np.array_equal(k,I) and (not np.array_equal(j,I)):
            b.append(experiment_result)

        if (not np.array_equal(j,I)) and (not np.array_equal(k,I)):
            T.append(experiment_result)

    return a,b,T

def computed_fidelity(a1,b1,T1,U):
    a2, b2, T2 = groundstate_representation(U)
    return np.real(0.25 * (1 + np.dot(a1,a2) + np.dot(b1,b2) + np.dot(T1,T2)))

def results(max_rounds):

    t = 2 #2 qubits per register
    x = range(1,max_rounds+1) #Maximum number of iterations: First iteration is PEA + Scram
    #Created using first two functions in code
    U = np.array([[ 9.41919642e-01+0.01406486j, -1.60666504e-01+0.11125395j,
      -2.63619647e-01-0.01450135j,  1.19179214e-02+0.0674593j ],
     [-1.60666504e-01+0.11125395j,  9.39269571e-03+0.87073143j,
      -9.09275978e-02-0.11386241j, -1.53863353e-03+0.42698072j],
     [-2.63619647e-01-0.01450135j, -9.09275978e-02-0.11386241j,
      -9.51449571e-01+0.01487471j,  3.93241074e-04-0.05986835j],
     [ 1.19179214e-02+0.0674593j,  -1.53863353e-03+0.42698072j,
       3.93241074e-04-0.05986835j,  1.37233533e-04-0.899671j]])

    with open('Groundstate_Preparation_Two_Qubit_Hamiltonian.csv', 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter="\t")
        writer.writerow(['Rounds', 'mu_1','mu_2', 'Fidelity with noise'])
        for p in tqdm(x):
            for mu1,mu2,sigma in zip([50e3,180e3,180e4],[70e3,200e3,200e4],[10e3,36e3,10e4]):
                a = []
                b = []
                T = []
                for j,k in itertools.product(['I','z','x','y'],['I','z','x','y']):

                    if j == 'I' and k == 'I':
                        continue

                    experiment_result = experiment(p,t,j,k,U,noise_model = thermal_noise(mu1,mu2,sigma,p,t))

                    if j =='I' and k != 'I':
                        a.append(experiment_result)
                    if k == 'I' and j != 'I':
                        b.append(experiment_result)
                    if j != 'I' and k != 'I':
                        T.append(experiment_result)

                writer.writerow([p,mu1,mu2,computed_fidelity(a,b,T,U)])


def plot_results(max_rounds):
    x = range(1,max_rounds+1)
    data = np.genfromtxt('Groundstate_Preparation_Two_Qubit_Hamiltonian.csv',delimiter="\t",skip_header=1)
    for i,k in zip(range(3),['High','Med','Low']):
        results_noise = [data[j][3] for j in range(i,len(data),3)]
        plt.plot(x,results_noise,label = '{}'.format(k))

    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
    plt.xlabel('$Number \ of \ Rounds$')
    plt.ylabel('$Fidelity$')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    max_rounds = 1
    results(max_rounds)
    plot_results(max_rounds)
