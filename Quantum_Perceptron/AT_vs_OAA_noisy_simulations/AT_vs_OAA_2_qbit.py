from qiskit import *
import matplotlib.pyplot as plt
import numpy as np
from qiskit.quantum_info.operators import Operator
from qiskit.tools.monitor import job_monitor
import csv
from qiskit.providers.aer.noise.errors import thermal_relaxation_error
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise import depolarizing_error
from qiskit.providers.aer import StatevectorSimulator
from qiskit.circuit.library import MCMT
from qiskit.circuit.library import OR
from qiskit.transpiler.passes.basis import Unroller
from qiskit.converters import circuit_to_dag, dag_to_circuit
from tqdm import tqdm

def minus_iY():

    """
    Minus iY Unitary
    """

    gate = Operator(np.array([[0,-1],[1,0]]))
    target = QuantumRegister(1,'t_qbit')
    qc = QuantumCircuit(target)
    qc.unitary(gate,[*target])
    return qc

def U(theta):

    """
    Post-select unitary U
    """
    register = QuantumRegister(2,'a_qbit')
    target = QuantumRegister(1,'t_qbit')

    qc = QuantumCircuit(register,target)

    qc.ry(2*theta,[*register])
    qc.append(MCMT(minus_iY(),2,1),[*register,*target])
    qc.ry(-2*theta,[*register])

    return qc.to_instruction()

def controlled_controlled_minus_iY():

    target = QuantumRegister(1,'t_qbit')
    register = QuantumRegister(2,'r_qbit')
    ancillae = QuantumRegister(1,'a_qbit')
    or_qubit = QuantumRegister(1,'o_qbit')
    qc = QuantumCircuit(or_qubit,register,ancillae,target)

    qc.ccx(or_qubit,register[0],ancillae[0])
    qc.append(MCMT(minus_iY(),2,1),[ancillae[-1],register[-1],target])
    qc.ccx(or_qubit,register[0],ancillae[0])

    return qc.to_instruction()

def controlled_U(theta):
    """
    Controlled post-select unitary U
    """
    new_qubits = QuantumRegister(2,'n_qbit')
    old_qubits = QuantumRegister(2,'o_qbit')
    control_qubits = QuantumRegister(1,'c_qbit')
    or_qubits = QuantumRegister(1,'o_qbit')
    target = QuantumRegister(1,'t_qbit')
    qc = QuantumCircuit(or_qubits,new_qubits,control_qubits,target)

    qc.ry(2*theta,[*new_qubits])
    qc.append(controlled_controlled_minus_iY(),[*or_qubits,*new_qubits,*control_qubits,*target])
    qc.ry(-2*theta,[*new_qubits])

    return qc.to_instruction()

def ancillae_thermalisation(theta,n,measure=None,gate_number = False,noise_model = None):

    register_qubits = QuantumRegister(2 + n*2 ,'r_qbit')
    target = QuantumRegister(1,'t_qbit')
    c = ClassicalRegister(1)

    if n == 0:
        circ = QuantumCircuit(register_qubits,target,c)
        circ.append(U(theta),[*register_qubits[:2],target])

    else:
        or_qubits = QuantumRegister(n,'o_qbit')
        control_qubits = QuantumRegister(1,'c_qbit')
        circ = QuantumCircuit(register_qubits,or_qubits,control_qubits,target,c)

        #Initial Round
        circ.append(U(theta),[*register_qubits[:2],target])

        #Subsequent Rounds
        for i in range(1,n+1):

            #OR Gate (for 2 register qubits)
            circ.append(OR(2),[*register_qubits[2*(i-1):2*i],or_qubits[i-1]])

            #W Gate
            circ.cry(np.pi/2,or_qubits[i-1],target)

            #U Gate
            circ.append(controlled_U(theta),[or_qubits[i-1],*register_qubits[2*i:2*(i+1)],*control_qubits,target])

    if gate_number:

        dag = circuit_to_dag(circ)
        pass_ = Unroller(['u3','cx']).run(dag)
        t_circ = dag_to_circuit(pass_)
        gate_ops = 0
        for instr, _, _ in t_circ:
            if instr.name not in ['barrier', 'snapshot'] and not instr.params == [0,0,0]:
                gate_ops += 1
        return gate_ops

    #Pauli Measurements
    if measure == 'x':
        circ.h(target)
    elif measure == 'y':
        circ.sdg(target)
        circ.h(target)

    circ.measure(target,c)

    shots = 8192

    #IBMQ.load_account()
    #provider = IBMQ.get_provider('ibm-q')

    if noise_model is None:
        #qcomp = provider.get_backend('ibmq_16_melbourne')
        #qcomp = provider.get_backend('ibmq_qasm_simulator')
        qcomp = Aer.get_backend('qasm_simulator')
        job = execute(circ, backend = qcomp, shots = shots)


    else:
        #qcomp = provider.get_backend('ibmq_qasm_simulator')
        qcomp = Aer.get_backend('qasm_simulator')
        q_num = 1 + (2 + 2*n + n + 1) #target, register, or qubits, control qubits
        noise_model = noise_model(q_num)
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

def oaa(theta,n,measure = None,gate_number = False,noise_model = None):

    def S_m():

        """
        Reflection operator S_m(pi/3)
        """

        def phase_shift():
            S = Operator(np.array([[1,0],[0,np.exp(1j*np.pi/3)]]))
            target = QuantumRegister(1,'t_qbit')
            qc = QuantumCircuit(target)
            qc.unitary(S,[*target])
            return qc


        target_register = QuantumRegister(1,'ta_qbit')
        register = QuantumRegister(1,'r_qbit')
        qc = QuantumCircuit(target_register,register)

        qc.x([*register,*target_register])
        qc.append(MCMT(phase_shift(),1,1),[*register,*target_register])
        qc.x([*register,*target_register])
        return qc.to_instruction()

    #Initial Round
    circuit = U(theta)

    if n == 0:
        register_qubits = QuantumRegister(2,'r_qbit')
        target = QuantumRegister(1,'t_qbit')
        c = ClassicalRegister(1)

        circ = QuantumCircuit(register_qubits,target,c)
        circ.append(circuit,[*register_qubits,*target])


    for i in range(1,n+1):
        register_qubits = QuantumRegister(2,'r_qbit')
        target = QuantumRegister(1,'t_qbit')
        c = ClassicalRegister(1)

        circ = QuantumCircuit(register_qubits,target,c)

        circ.append(circuit,[*register_qubits,*target])
        circ.append(S_m(),[*register_qubits])
        circ.append(circuit.inverse(),[*register_qubits,*target])
        circ.append(S_m(),[*register_qubits])
        circ.append(circuit,[*register_qubits,*target])

        circuit = circ.to_instruction()

    if gate_number:

        dag = circuit_to_dag(circ)
        pass_ = Unroller(['u3','cx']).run(dag)
        t_circ = dag_to_circuit(pass_)
        gate_ops = 0
        for instr, _, _ in t_circ:
            if instr.name not in ['barrier', 'snapshot'] and not instr.params == [0,0,0]:
                gate_ops += 1
        return gate_ops

    #Pauli Measurements
    if measure == 'x':
        circ.h(target)

    elif measure == 'y':
        circ.sdg(target)
        circ.h(target)

    circ.measure(target,c)

    shots = 8192

    #IBMQ.load_account()
    #provider = IBMQ.get_provider('ibm-q')

    if noise_model is None:
        #qcomp = provider.get_backend('ibmq_16_melbourne')
        #qcomp = provider.get_backend('ibmq_qasm_simulator')
        qcomp = Aer.get_backend('qasm_simulator')
        job = execute(circ, backend = qcomp, shots = shots)

    else:
        #qcomp = provider.get_backend('ibmq_qasm_simulator')
        qcomp = Aer.get_backend('qasm_simulator')

        noise_model = noise_model(2+1)
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

def thermal_noise(n_qubits,mu1,mu2,sigma):
    num = n_qubits
    # T1 and T2 values for qubits 0-num
    T1s = np.random.normal(mu1, sigma, num) # Sampled from normal distribution mean mu1 microsec
    T2s = np.random.normal(mu2, sigma, num)  # Sampled from normal distribution mean mu2 microsec

    # Truncate random T2s <= T1s
    T2s = np.array([min(T2s[j], 2 * T1s[j]) for j in range(num)])

    # Instruction times (in nanoseconds)
    time_u1 = 0   # virtual gate
    time_u2 = 50  # (single X90 pulse)
    time_u3 = 100 # (two X90 pulses)
    time_cx = 300

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

    # Add errors to noise model
    noise_thermal = NoiseModel()

    for j in range(num):
        noise_thermal.add_quantum_error(errors_u1[j], "u1", [j])
        noise_thermal.add_quantum_error(errors_u2[j], "u2", [j])
        noise_thermal.add_quantum_error(errors_u3[j], "u3", [j])
        for k in range(num):
            noise_thermal.add_quantum_error(errors_cx[j][k], "cx", [j, k])

    return noise_thermal

def depolarizing_noise(n_qubits,p1,p2):

    # Depolarizing quantum errors
    error_1 = depolarizing_error(p1, 1)
    error_2 = depolarizing_error(p2, 2)

    # Add errors to noise model
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(error_1, ['u1', 'u2', 'u3'])
    noise_model.add_all_qubit_quantum_error(error_2, ['cx'])

    return noise_model

def experiment(theta,noise_model = None):

    with open('oaa_thermal_noise_two.csv', 'w') as csvfile:

        writer = csv.writer(csvfile, delimiter="\t")
        writer.writerow(['Theta','num_rounds','Gates','<X>', '<Y>', '<Z>'])

        for num_rounds in tqdm(range(0,3)):
            X_exp = oaa(theta,num_rounds,measure = 'x',noise_model = noise_model)
            Y_exp = oaa(theta,num_rounds,measure = 'y',noise_model = noise_model)
            Z_exp = oaa(theta,num_rounds,measure = 'z',noise_model = noise_model)
            gate_num = oaa(theta,num_rounds,gate_number = True)

            writer.writerow([theta,num_rounds,gate_num,X_exp,Y_exp,Z_exp])

    with open('ancillae_thermalisation_thermal_noise_two.csv', 'w') as csvfile:

        writer = csv.writer(csvfile, delimiter="\t")
        writer.writerow(['Theta','num_rounds','Gates','<X>', '<Y>', '<Z>'])

        for num_rounds in tqdm(range(0,3)):
            X_exp = ancillae_thermalisation(theta,num_rounds,measure = 'x',noise_model = noise_model)
            Y_exp = ancillae_thermalisation(theta,num_rounds,measure = 'y',noise_model = noise_model)
            Z_exp = ancillae_thermalisation(theta,num_rounds,measure = 'z',noise_model = noise_model)
            gate_num = ancillae_thermalisation(theta,num_rounds,gate_number = True)

            writer.writerow([theta,num_rounds,gate_num,X_exp,Y_exp,Z_exp])

def results():
    def p(theta):
        return np.cos(theta)**4 + np.sin(theta)**4

    def computed_fidelity(theta,X_exp,Y_exp,Z_exp):
        computed_list = np.array([X_exp,Y_exp,Z_exp])
        expected_list = np.array([2*np.cos(theta)**2*np.sin(theta)**2, 0, np.cos(theta)**4-np.sin(theta)**4])/p(theta)
        return 0.5*(1+np.dot(computed_list, expected_list))


    data = np.genfromtxt('ancillae_thermalisation_thermal_noise_two.csv', delimiter="\t",skip_header=1)
    theta = np.arcsin(np.sin(data[0,0])**2)
    fidelities = [computed_fidelity(theta,*data[i,3:]) for i in range(len(data))]
    num_ops = [data[i,2] for i in range(len(data))]
    plt.plot(num_ops,fidelities,label = 'AT High Noise',color = 'blue')
    plt.xlim([min(num_ops),max(num_ops)])

    data = np.genfromtxt('oaa_thermal_noise_two.csv', delimiter="\t",skip_header=1)
    theta = np.arcsin(np.sin(data[0,0])**2)
    fidelities = [computed_fidelity(theta,*data[i,3:]) for i in range(len(data))]
    num_ops = [data[i,2] for i in range(len(data))]
    plt.plot(num_ops,fidelities,label = 'OAA High Noise',color = 'orange')

    plt.xlabel('Number of Operations')
    plt.ylabel('Fidelity')
    plt.title('Fidelity between the finalised target state and desired state for the \nQuantum Gearbox, 2 qubits,with dephasing noise')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    theta = np.pi/4


    params = [50e3,70e3,10e3]
    noise_model = lambda q_num: thermal_noise(q_num,*params)

    #params = [0.001,0.01]
    #noise_model = lambda q_num: depolarizing_noise(q_num,*params)

    #noise_model = None

    experiment(theta,noise_model = noise_model)
    results()
