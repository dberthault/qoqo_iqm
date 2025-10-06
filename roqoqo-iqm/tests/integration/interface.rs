// Copyright © 2020-2023 HQS Quantum Simulations GmbH. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the
// License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
// express or implied. See the License for the specific language governing permissions and
// limitations under the License.

use ndarray::Array1;
use num_complex::Complex64;
use qoqo_calculator::{Calculator, CalculatorFloat};
use rand::distr::{Distribution, StandardUniform, Uniform};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use roqoqo::{backends::EvaluatingBackend, operations::*, Circuit, RoqoqoBackendError};
use roqoqo_iqm::{
    call_circuit, call_operation, virtual_z_replacement_circuit, IqmBackendError, IqmCircuit,
    IqmInstruction,
};

use std::collections::HashMap;
use std::f64::consts::PI;
use test_case::test_case;

const TEST_TOLERANCE: f64 = 1e-9;

#[test_case(
    RotateXY::new(1, PI.into(), PI.into()).into(),
    IqmInstruction {
        name : "prx".to_string(),
        qubits: vec!["QB2".to_string()],
        args : HashMap::from([
            ("angle_t".to_string(), CalculatorFloat::Float(0.5)),
            ("phase_t".to_string(), CalculatorFloat::Float(0.5))
        ]),
    };
    "Phased X Rotation")]
#[test_case(
        ControlledPauliZ::new(1, 2).into(),
        IqmInstruction {
            name : "cz".to_string(),
            qubits: vec!["QB2".to_string(), "QB3".to_string()],
            args: HashMap::new(),
        };
        "Controlled Z")]
#[test_case(
    CZQubitResonator::new(1, 0).into(),
    IqmInstruction {
        name : "cz".to_string(),
        qubits: vec!["QB2".to_string(), "COMP_R".to_string()],
        args: HashMap::new(),
    };
    "CZQubitResonator")]
#[test_case(
    SingleExcitationLoad::new(5, 0).into(),
    IqmInstruction {
        name : "move".to_string(),
        qubits: vec!["QB6".to_string(), "COMP_R".to_string()],
        args: HashMap::new(),
    };
    "SingleExcitationLoad")]
#[test_case(
    SingleExcitationStore::new(5, 0).into(),
    IqmInstruction {
        name : "move".to_string(),
        qubits: vec!["QB6".to_string(), "COMP_R".to_string()],
        args: HashMap::new(),
    };
    "SingleExcitationStore")]
fn test_passing_interface(operation: Operation, instruction: IqmInstruction) {
    let called = call_operation(&operation).unwrap().unwrap();
    assert_eq!(instruction, called);
}

#[test_case(CNOT::new(0, 1).into(); "CNOT")]
#[test_case(RotateX::new(0, 1.0.into()).into(); "RotateX")]
#[test_case(Hadamard::new(0).into(); "Hadamard")]
fn test_failure_unsupported_operation(operation: Operation) {
    let called = call_operation(&operation);
    match called {
        Err(RoqoqoBackendError::OperationNotInBackend { .. }) => {}
        _ => panic!("Not the right error"),
    }
}

#[test]
fn test_call_circuit_single_measurement() {
    let mut circuit = Circuit::new();
    let register_length = 3;
    let readout_name = "ro".to_string();
    circuit += ControlledPauliZ::new(0, 1);
    circuit += RotateXY::new(0, PI.into(), PI.into());
    circuit += DefinitionBit::new(readout_name.clone(), register_length, true);
    circuit += MeasureQubit::new(0, readout_name.clone(), 0);
    circuit += MeasureQubit::new(1, readout_name.clone(), 1);
    let res = call_circuit(circuit.iter(), 2, None, 1).unwrap().0;

    let cz_instruction = IqmInstruction {
        name: "cz".to_string(),
        qubits: vec!["QB1".to_string(), "QB2".to_string()],
        args: HashMap::new(),
    };
    let xy_instruction = IqmInstruction {
        name: "prx".to_string(),
        qubits: vec!["QB1".to_string()],
        args: HashMap::from([
            ("angle_t".to_string(), CalculatorFloat::Float(0.5)),
            ("phase_t".to_string(), CalculatorFloat::Float(0.5)),
        ]),
    };
    let meas_instruction = IqmInstruction {
        name: "measure".to_string(),
        qubits: vec!["QB1".to_string(), "QB2".to_string()],
        args: HashMap::from([(
            "key".to_string(),
            CalculatorFloat::Str(readout_name.clone()),
        )]),
    };
    let instruction_vec = vec![cz_instruction, xy_instruction, meas_instruction];

    let mut metadata = HashMap::new();
    metadata.insert(readout_name, (vec![0, 1], register_length));

    let res_expected: IqmCircuit = IqmCircuit {
        name: String::from("qc_1"),
        instructions: instruction_vec,
        metadata: Some(metadata),
    };

    assert_eq!(res, res_expected)
}

#[test]
fn test_call_circuit_single_measurement_load_store() {
    let mut circuit = Circuit::new();
    let register_length = 3;
    let readout_name = "ro".to_string();
    circuit += ControlledPauliZ::new(0, 1);
    circuit += RotateXY::new(0, PI.into(), PI.into());
    circuit += SingleExcitationStore::new(3, 0);
    circuit += SingleExcitationLoad::new(3, 0);
    circuit += DefinitionBit::new(readout_name.clone(), register_length, true);
    circuit += MeasureQubit::new(0, readout_name.clone(), 0);
    circuit += MeasureQubit::new(1, readout_name.clone(), 1);
    let res = call_circuit(circuit.iter(), 2, None, 1).unwrap().0;

    let cz_instruction = IqmInstruction {
        name: "cz".to_string(),
        qubits: vec!["QB1".to_string(), "QB2".to_string()],
        args: HashMap::new(),
    };
    let xy_instruction = IqmInstruction {
        name: "prx".to_string(),
        qubits: vec!["QB1".to_string()],
        args: HashMap::from([
            ("angle_t".to_string(), CalculatorFloat::Float(0.5)),
            ("phase_t".to_string(), CalculatorFloat::Float(0.5)),
        ]),
    };
    let load_instruction = IqmInstruction {
        name: "move".to_string(),
        qubits: vec!["QB4".to_string(), "COMP_R".to_string()],
        args: HashMap::new(),
    };
    let store_instruction = IqmInstruction {
        name: "move".to_string(),
        qubits: vec!["QB4".to_string(), "COMP_R".to_string()],
        args: HashMap::new(),
    };
    let meas_instruction = IqmInstruction {
        name: "measure".to_string(),
        qubits: vec!["QB1".to_string(), "QB2".to_string()],
        args: HashMap::from([(
            "key".to_string(),
            CalculatorFloat::Str(readout_name.clone()),
        )]),
    };
    let instruction_vec = vec![
        cz_instruction,
        xy_instruction,
        load_instruction,
        store_instruction,
        meas_instruction,
    ];
    let mut metadata = HashMap::new();
    metadata.insert(readout_name, (vec![0, 1], register_length));

    let res_expected: IqmCircuit = IqmCircuit {
        name: String::from("qc_1"),
        instructions: instruction_vec,
        metadata: Some(metadata),
    };

    assert_eq!(res, res_expected)
}

#[test]
fn test_call_circuit_repeated_measurement_passes() {
    let mut inner_circuit = Circuit::new();
    inner_circuit += ControlledPauliZ::new(0, 1);

    let mut circuit = Circuit::new();
    let register_length = 2;
    let readout_name = "ro".to_string();
    let number_measurements_expected = 100;
    circuit += ControlledPauliZ::new(0, 1);
    circuit += RotateXY::new(0, PI.into(), PI.into());
    circuit += CZQubitResonator::new(1, 0);
    circuit += SingleExcitationStore::new(5, 0);
    circuit += SingleExcitationLoad::new(5, 0);
    circuit += PragmaLoop::new(CalculatorFloat::Float(3.0), inner_circuit);
    circuit += DefinitionBit::new(readout_name.clone(), register_length, true);
    circuit += MeasureQubit::new(0, readout_name.clone(), 0);
    circuit += MeasureQubit::new(1, readout_name.clone(), 1);
    circuit +=
        PragmaSetNumberOfMeasurements::new(number_measurements_expected, readout_name.clone());

    let cz_instruction = IqmInstruction {
        name: "cz".to_string(),
        qubits: vec!["QB1".to_string(), "QB2".to_string()],
        args: HashMap::new(),
    };
    let xy_instruction = IqmInstruction {
        name: "prx".to_string(),
        qubits: vec!["QB1".to_string()],
        args: HashMap::from([
            ("angle_t".to_string(), CalculatorFloat::Float(0.5)),
            ("phase_t".to_string(), CalculatorFloat::Float(0.5)),
        ]),
    };
    let cz_qubit_resonator_instruction = IqmInstruction {
        name: "cz".to_string(),
        qubits: vec!["QB2".to_string(), "COMP_R".to_string()],
        args: HashMap::new(),
    };
    let load_instruction = IqmInstruction {
        name: "move".to_string(),
        qubits: vec!["QB6".to_string(), "COMP_R".to_string()],
        args: HashMap::new(),
    };
    let store_instruction = IqmInstruction {
        name: "move".to_string(),
        qubits: vec!["QB6".to_string(), "COMP_R".to_string()],
        args: HashMap::new(),
    };
    let meas_instruction = IqmInstruction {
        name: "measure".to_string(),
        qubits: vec!["QB1".to_string(), "QB2".to_string()],
        args: HashMap::from([(
            "key".to_string(),
            CalculatorFloat::Str(readout_name.clone()),
        )]),
    };
    let mut instruction_vec = vec![
        cz_instruction.clone(),
        xy_instruction,
        cz_qubit_resonator_instruction,
        load_instruction,
        store_instruction,
    ];
    for _ in 0..3 {
        instruction_vec.push(cz_instruction.clone());
    }
    instruction_vec.push(meas_instruction);

    let mut metadata = HashMap::new();
    metadata.insert(readout_name, (vec![0, 1], register_length));

    let res_expected: IqmCircuit = IqmCircuit {
        name: String::from("qc_1"),
        instructions: instruction_vec,
        metadata: Some(metadata),
    };
    let (res, number_measurements) = call_circuit(circuit.iter(), 2, None, 1).unwrap();

    assert_eq!(res, res_expected);
    assert_eq!(number_measurements, number_measurements_expected);
}

// test that setting multiple measurements with different numbers of measurements throws an error
#[test]
fn test_call_circuit_repeated_measurement_error() {
    let number_measurements_1 = 10;
    let number_measurements_2 = 20;

    let mut circuit = Circuit::new();

    circuit += RotateXY::new(2, 1.0.into(), 1.0.into());
    circuit += CZQubitResonator::new(1, 0);
    circuit += SingleExcitationStore::new(5, 0);
    circuit += SingleExcitationLoad::new(5, 0);
    circuit += DefinitionBit::new("reg1".to_string(), 5, true);
    circuit += DefinitionBit::new("reg2".to_string(), 7, true);
    circuit += MeasureQubit::new(2, "reg1".to_string(), 2);
    circuit += MeasureQubit::new(3, "reg1".to_string(), 3);
    circuit += MeasureQubit::new(1, "reg2".to_string(), 1);
    circuit += PragmaSetNumberOfMeasurements::new(number_measurements_1, "reg1".to_string());
    circuit += PragmaSetNumberOfMeasurements::new(number_measurements_2, "reg2".to_string());

    let err = call_circuit(circuit.iter(), 6, None, 1);
    assert!(matches!(err, Err(IqmBackendError::InvalidCircuit { .. })))
}

// test the an error is returned when a measurement operation tries to write to an undefined register
#[test]
fn test_call_circuit_undefined_register_error() {
    let mut circuit = Circuit::new();

    circuit += DefinitionBit::new("reg1".to_string(), 5, true);
    circuit += RotateXY::new(2, 1.0.into(), 1.0.into());
    circuit += CZQubitResonator::new(1, 0);
    circuit += SingleExcitationStore::new(5, 0);
    circuit += SingleExcitationLoad::new(5, 0);
    circuit += MeasureQubit::new(2, "reg2".to_string(), 2);

    let err = call_circuit(circuit.iter(), 6, None, 1);
    assert!(matches!(err, Err(IqmBackendError::InvalidCircuit { .. })));

    circuit += DefinitionBit::new("reg1".to_string(), 5, true);
    circuit += RotateXY::new(2, 1.0.into(), 1.0.into());
    circuit += CZQubitResonator::new(1, 0);
    circuit += SingleExcitationStore::new(5, 0);
    circuit += SingleExcitationLoad::new(5, 0);
    circuit += MeasureQubit::new(2, "reg1".to_string(), 2);
    circuit += PragmaSetNumberOfMeasurements::new(10, "reg2".to_string());

    let err = call_circuit(circuit.iter(), 6, None, 1);
    assert!(matches!(err, Err(IqmBackendError::InvalidCircuit { .. })));
}

// test the an error is returned when a qubit is being measured twice
#[test]
fn test_symbolic_pragma_loop_error() {
    let mut inner_circuit = Circuit::new();
    inner_circuit += RotateXY::new(2, 1.0.into(), 1.0.into());

    let mut circuit = Circuit::new();
    circuit += DefinitionBit::new("reg1".to_string(), 5, true);
    circuit += CZQubitResonator::new(1, 0);
    circuit += SingleExcitationStore::new(5, 0);
    circuit += SingleExcitationLoad::new(5, 0);

    circuit += PragmaLoop::new("repetitions".into(), inner_circuit);

    circuit += MeasureQubit::new(2, "reg1".to_string(), 2);
    circuit += MeasureQubit::new(2, "reg1".to_string(), 2);

    let err = call_circuit(circuit.iter(), 6, None, 1);
    assert!(matches!(err, Err(IqmBackendError::InvalidCircuit { .. })))
}

// test the an error is returned when a qubit is being measured twice
#[test]
fn test_qubit_measured_twice_error() {
    let mut circuit = Circuit::new();

    circuit += DefinitionBit::new("reg1".to_string(), 5, true);
    circuit += RotateXY::new(2, 1.0.into(), 1.0.into());
    circuit += CZQubitResonator::new(1, 0);
    circuit += SingleExcitationStore::new(5, 0);
    circuit += SingleExcitationLoad::new(5, 0);
    circuit += MeasureQubit::new(2, "reg1".to_string(), 2);
    circuit += MeasureQubit::new(2, "reg1".to_string(), 2);

    let err = call_circuit(circuit.iter(), 6, None, 1);
    assert!(matches!(err, Err(IqmBackendError::InvalidCircuit { .. })))
}

#[test]
fn test_call_circuit_repeated_measurements_with_mapping() {
    let mut circuit = Circuit::new();
    circuit += ControlledPauliZ::new(0, 1);
    circuit += RotateXY::new(0, 1.0.into(), 1.0.into());
    circuit += DefinitionBit::new("ro".to_string(), 2, true);
    let qubit_mapping = HashMap::from([(0, 1), (1, 0)]);
    circuit += PragmaRepeatedMeasurement::new("ro".to_string(), 3, Some(qubit_mapping));
    let ok = call_circuit(circuit.iter(), 2, None, 1).is_ok();

    assert!(ok);
}

#[test]
fn test_fail_multiple_repeated_measurements() {
    let mut circuit = Circuit::new();
    circuit += ControlledPauliZ::new(0, 1);
    circuit += DefinitionBit::new("ro".to_string(), 2, true);
    circuit += PragmaSetNumberOfMeasurements::new(5, "ro".to_string());
    circuit += PragmaRepeatedMeasurement::new("ro".to_string(), 3, None);
    let res = call_circuit(circuit.iter(), 2, None, 1);

    assert!(res.is_err());
}

#[test]
fn test_fail_overlapping_measurements() {
    let mut circuit = Circuit::new();
    circuit += ControlledPauliZ::new(0, 1);
    circuit += DefinitionBit::new("ro".to_string(), 2, true);
    circuit += MeasureQubit::new(0, "ro".to_string(), 0);
    circuit += PragmaRepeatedMeasurement::new("ro".to_string(), 3, None);
    let res = call_circuit(circuit.iter(), 2, None, 1);

    assert!(res.is_err());
}

fn compare_circuit_helper(left: &Circuit, right: &Circuit) {
    assert_eq!(left.len(), right.len());
    for (op_left, op_right) in left.iter().zip(right.iter()) {
        if let Ok(op_sq_left) = SingleQubitGateOperation::try_from(op_left) {
            let op_sq_right = SingleQubitGateOperation::try_from(op_right).unwrap();
            if let Ok(some) = op_sq_left.alpha_r().float() {
                assert!((some - op_sq_right.alpha_r().float().unwrap()).abs() < TEST_TOLERANCE);
            } else {
                assert_eq!(op_sq_left.alpha_r(), op_sq_right.alpha_r());
            }
            if let Ok(some) = op_sq_left.alpha_i().float() {
                assert!((some - op_sq_right.alpha_i().float().unwrap()).abs() < TEST_TOLERANCE);
            } else {
                assert_eq!(op_sq_left.alpha_i(), op_sq_right.alpha_i());
            }
            if let Ok(some) = op_sq_left.beta_r().float() {
                assert!((some - op_sq_right.beta_r().float().unwrap()).abs() < TEST_TOLERANCE);
            } else {
                assert_eq!(op_sq_left.beta_r(), op_sq_right.beta_r());
            }
            if let Ok(some) = op_sq_left.beta_i().float() {
                assert!((some - op_sq_right.beta_i().float().unwrap()).abs() < TEST_TOLERANCE);
            } else {
                assert_eq!(op_sq_left.beta_i(), op_sq_right.beta_i());
            }
        } else {
            assert_eq!(op_left, op_right);
        }
    }
}

fn construct_random_circuit(circuit_length: usize, number_qubits: usize, seed: u64) -> Circuit {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut circuit = Circuit::new();
    for _ in 0..circuit_length {
        let tmp_seed: u64 = rng.sample(StandardUniform);
        add_random_operation(&mut circuit, number_qubits, tmp_seed)
    }
    circuit
}

fn add_random_operation(circuit: &mut Circuit, number_qubits: usize, seed: u64) {
    let mut rng = StdRng::seed_from_u64(seed);
    let qubits_dist = Uniform::new(0, number_qubits).unwrap();
    let two_qubits_dist = Uniform::new(0, number_qubits - 1).unwrap();
    let gate_type_dist = Uniform::new(0, 16).unwrap();
    let new_op: Operation = match gate_type_dist.sample(&mut rng) {
        0 => PauliX::new(qubits_dist.sample(&mut rng)).into(),
        1 => PauliY::new(qubits_dist.sample(&mut rng)).into(),
        2 => PauliZ::new(qubits_dist.sample(&mut rng)).into(),
        4 => SqrtPauliX::new(qubits_dist.sample(&mut rng)).into(),
        5 => InvSqrtPauliX::new(qubits_dist.sample(&mut rng)).into(),
        6 => {
            let theta: f64 = rng.sample(StandardUniform);
            RotateX::new(qubits_dist.sample(&mut rng), theta.into()).into()
        }
        7 => {
            let theta: f64 = rng.sample(StandardUniform);
            RotateY::new(qubits_dist.sample(&mut rng), theta.into()).into()
        }
        8 => {
            let theta: f64 = rng.sample(StandardUniform);
            RotateZ::new(qubits_dist.sample(&mut rng), theta.into()).into()
        }

        9 => {
            let qubit = two_qubits_dist.sample(&mut rng);
            ControlledPauliZ::new(qubit, qubit + 1).into()
        }

        10 => {
            let theta: f64 = rng.sample(StandardUniform);
            let qubit = two_qubits_dist.sample(&mut rng);
            ControlledPhaseShift::new(qubit, qubit + 1, theta.into()).into()
        }
        11 => {
            let theta: f64 = rng.sample(StandardUniform);
            let qubit = qubits_dist.sample(&mut rng);
            PhaseShiftState0::new(qubit, theta.into()).into()
        }
        12 => {
            let theta: f64 = rng.sample(StandardUniform);
            let qubit = qubits_dist.sample(&mut rng);
            PhaseShiftState1::new(qubit, theta.into()).into()
        }
        13 => {
            let phi: f64 = rng.sample(StandardUniform);
            let qubit = two_qubits_dist.sample(&mut rng);
            PhaseShiftedControlledZ::new(qubit, qubit + 1, phi.into()).into()
        }
        14 => {
            let phi: f64 = rng.sample(StandardUniform);
            let theta: f64 = rng.sample(StandardUniform);
            let qubit = two_qubits_dist.sample(&mut rng);
            PhaseShiftedControlledPhase::new(qubit, qubit + 1, phi.into(), theta.into()).into()
        }
        15 => {
            let theta: f64 = rng.sample(StandardUniform);
            let phi: f64 = rng.sample(StandardUniform);
            RotateXY::new(qubits_dist.sample(&mut rng), theta.into(), phi.into()).into()
        }
        _ => {
            let theta: f64 = rng.sample(StandardUniform);
            RotateZ::new(qubits_dist.sample(&mut rng), theta.into()).into()
        }
    };
    circuit.add_operation(new_op);
}

#[test]
fn test_simplify_rotate_z() {
    let mut circuit = Circuit::new();
    circuit += RotateZ::new(0, 0.1.into());
    circuit += RotateZ::new(0, (-0.2).into());
    circuit = virtual_z_replacement_circuit(&circuit, None, true)
        .unwrap()
        .0;
    let mut test_circuit = Circuit::new();
    test_circuit += RotateZ::new(0, (-0.1).into()).to_single_qubit_gate();
    compare_circuit_helper(&circuit, &test_circuit);
}

#[test]
fn test_simplify_rotate_z_struct() {
    let mut circuit = Circuit::new();
    circuit += RotateZ::new(0, 0.1.into());
    circuit += RotateZ::new(0, (-0.2).into());
    circuit = virtual_z_replacement_circuit(&circuit, None, true)
        .unwrap()
        .0;
    let mut test_circuit = Circuit::new();
    test_circuit += RotateZ::new(0, (-0.1).into()).to_single_qubit_gate();
    compare_circuit_helper(&circuit, &test_circuit);

    circuit += RotateZ::new(0, 0.1.into());
    circuit += RotateZ::new(0, (-0.2).into());
    circuit = virtual_z_replacement_circuit(&circuit, None, false)
        .unwrap()
        .0;
    let test_circuit = Circuit::new();
    compare_circuit_helper(&circuit, &test_circuit);
}

#[test]
fn test_simplify_rotate_x_rotate_z() {
    let mut circuit = Circuit::new();
    circuit += RotateZ::new(0, 0.1.into());
    circuit += RotateX::new(0, 0.3.into());
    circuit += RotateZ::new(0, (-0.2).into());
    let circuit = virtual_z_replacement_circuit(&circuit, None, true)
        .unwrap()
        .0;
    let mut test_circuit = Circuit::new();
    test_circuit += RotateXY::new(0, 0.3.into(), (-0.1).into());
    test_circuit += RotateZ::new(0, (-0.1).into()).to_single_qubit_gate();
    compare_circuit_helper(&circuit, &test_circuit);
}

#[test]
fn test_fails() {
    let mut circuit = Circuit::new();
    circuit += Hadamard::new(0);
    let res = virtual_z_replacement_circuit(&circuit, None, true);
    assert!(res.is_err());
    let mut circuit = Circuit::new();
    circuit += CNOT::new(0, 1);
    let res = virtual_z_replacement_circuit(&circuit, None, true);
    assert!(res.is_err());
    let mut circuit = Circuit::new();
    circuit += MultiQubitMS::new(vec![0, 1, 2], 0.1.into());
    let res = virtual_z_replacement_circuit(&circuit, None, true);
    assert!(res.is_err());
}

#[test]
fn test_pragma_repeated_measurement() {
    let number_qubits = 2;
    let mut circuit = Circuit::new();
    circuit += RotateZ::new(0, CalculatorFloat::from(0.567));
    circuit += PauliX::new(0);
    let mut trafo_circuit = circuit.clone() + DefinitionBit::new("ro_trafo".to_string(), 2, true);
    circuit += DefinitionBit::new("ro".to_string(), 2, true);
    circuit += PragmaRepeatedMeasurement::new("ro".to_string(), 30, None);
    trafo_circuit += PragmaRepeatedMeasurement::new("ro_trafo".to_string(), 30, None);
    let trafo_circuit = virtual_z_replacement_circuit(&trafo_circuit, None, true)
        .unwrap()
        .0;

    let backend = roqoqo_quest::Backend::new(number_qubits, Some(vec![2, 4]));
    let (circuit_result, _, _) = backend.run_circuit(&circuit).unwrap();
    let (circuit_transformed_result, _, _) = backend.run_circuit(&trafo_circuit).unwrap();
    let measurements = circuit_result.get("ro").unwrap()[0].clone();
    let measurements_trafo = circuit_transformed_result.get("ro_trafo").unwrap()[0].clone();
    assert_eq!(measurements, measurements_trafo);
}

#[test]
fn test_special_ops_variable_loop() {
    let number_qubits = 3;
    let dimension = 2_usize.pow(number_qubits as u32);
    let mut circuit = Circuit::new();
    circuit += RotateZ::new(0, CalculatorFloat::FRAC_PI_2);
    circuit += RotateX::new(0, CalculatorFloat::FRAC_PI_2);
    circuit += PragmaActiveReset::new(0);
    let mut inner_circuit = Circuit::new();
    inner_circuit += RotateX::new(1, 0.123.into());
    inner_circuit += RotateZ::new(1, 0.245.into());
    inner_circuit += RotateX::new(1, 0.678.into());
    inner_circuit += RotateX::new(0, 0.998.into());
    circuit += PragmaLoop::new("test".into(), inner_circuit.clone());
    circuit += RotateZ::new(0, 0.5.into());
    circuit += PauliZ::new(2);
    circuit += PauliX::new(2);
    circuit += DefinitionBit::new("ro_test".to_string(), 3, true);
    circuit += MeasureQubit::new(2, "ro_test".to_string(), 2);
    circuit += PragmaConditional::new("ro_test".to_string(), 2, inner_circuit);
    circuit += RotateZ::new(0, 0.5.into());
    let mut circuit_trafo =
        circuit.clone() + DefinitionComplex::new("ro_trafo".to_string(), dimension.pow(2), true);
    circuit += DefinitionComplex::new("ro".to_string(), dimension.pow(2), true);

    circuit += PragmaGetDensityMatrix::new("ro".to_string(), None);
    circuit_trafo += PragmaGetDensityMatrix::new("ro_trafo".to_string(), None);
    let mut calculator = Calculator::new();
    calculator.set_variable("test", 3.0);
    let transformed_circuit = virtual_z_replacement_circuit(&circuit_trafo, None, true)
        .unwrap()
        .0;
    for i in 0..dimension {
        let mut initial_statevector: Array1<Complex64> = Array1::zeros(dimension);
        initial_statevector[i] = Complex64::new(1.0, 0.0);
        let mut tmp_circuit = Circuit::new();
        let mut tmp_transformed_circuit = Circuit::new();

        tmp_circuit += PragmaSetStateVector::new(initial_statevector.clone());
        tmp_transformed_circuit += PragmaSetStateVector::new(initial_statevector.clone());
        tmp_circuit += circuit.substitute_parameters(&calculator).unwrap().clone();
        tmp_transformed_circuit += transformed_circuit
            .substitute_parameters(&calculator)
            .unwrap()
            .clone();

        let backend = roqoqo_quest::Backend::new(number_qubits, Some(vec![2, 4]));
        let (_, _, circuit_result) = backend.run_circuit(&tmp_circuit).unwrap();
        let (_, _, circuit_transformed_result) =
            backend.run_circuit(&tmp_transformed_circuit).unwrap();
        let final_statevector = circuit_result.get("ro").unwrap()[0].clone();
        let final_statevector_trafo =
            circuit_transformed_result.get("ro_trafo").unwrap()[0].clone();
        assert_eq!(final_statevector.len(), final_statevector_trafo.len());

        let (direct_val, transformed_val) = final_statevector
            .iter()
            .zip(final_statevector_trafo.iter())
            .find(|(x, _)| x.norm() > 1e-6)
            .unwrap();
        let global_phase_factor = transformed_val / direct_val;
        for (direct_val, transformed_val) in
            final_statevector.iter().zip(final_statevector_trafo.iter())
        {
            if (global_phase_factor * direct_val - transformed_val).norm() > 1e-6 {
                println!(
                    "direct val {}  transformed val {} difference norm {}, global phase factor {}",
                    direct_val,
                    transformed_val,
                    (direct_val - transformed_val).norm(),
                    global_phase_factor
                );
                panic!();
            }
        }
    }
}

#[test]
fn test_special_ops() {
    let number_qubits = 3;
    let dimension = 2_usize.pow(number_qubits as u32);
    let mut circuit = Circuit::new();
    circuit += RotateZ::new(0, CalculatorFloat::FRAC_PI_2);
    circuit += RotateX::new(0, CalculatorFloat::FRAC_PI_2);
    circuit += PragmaActiveReset::new(0);
    let mut inner_circuit = Circuit::new();
    inner_circuit += RotateX::new(1, 0.123.into());
    inner_circuit += RotateZ::new(1, 0.245.into());
    inner_circuit += RotateX::new(1, 0.678.into());
    inner_circuit += RotateX::new(0, 0.998.into());
    circuit += PragmaLoop::new(3.0.into(), inner_circuit.clone());
    circuit += RotateZ::new(0, 0.5.into());
    circuit += PauliZ::new(2);
    circuit += PauliX::new(2);
    circuit += DefinitionBit::new("ro_test".to_string(), 3, true);
    circuit += MeasureQubit::new(2, "ro_test".to_string(), 2);
    circuit += PragmaConditional::new("ro_test".to_string(), 2, inner_circuit);
    circuit += RotateZ::new(0, 0.5.into());
    let mut circuit_trafo =
        circuit.clone() + DefinitionComplex::new("ro_trafo".to_string(), dimension.pow(2), true);
    circuit += DefinitionComplex::new("ro".to_string(), dimension.pow(2), true);

    circuit += PragmaGetDensityMatrix::new("ro".to_string(), None);
    circuit_trafo += PragmaGetDensityMatrix::new("ro_trafo".to_string(), None);

    let transformed_circuit = virtual_z_replacement_circuit(&circuit_trafo, None, true)
        .unwrap()
        .0;
    for i in 0..dimension {
        let mut initial_statevector: Array1<Complex64> = Array1::zeros(dimension);
        initial_statevector[i] = Complex64::new(1.0, 0.0);
        let mut tmp_circuit = Circuit::new();
        let mut tmp_transformed_circuit = Circuit::new();

        tmp_circuit += PragmaSetStateVector::new(initial_statevector.clone());
        tmp_transformed_circuit += PragmaSetStateVector::new(initial_statevector.clone());
        tmp_circuit += circuit.clone();
        tmp_transformed_circuit += transformed_circuit.clone();

        let backend = roqoqo_quest::Backend::new(number_qubits, Some(vec![2, 4]));
        let (_, _, circuit_result) = backend.run_circuit(&tmp_circuit).unwrap();
        let (_, _, circuit_transformed_result) =
            backend.run_circuit(&tmp_transformed_circuit).unwrap();
        let final_statevector = circuit_result.get("ro").unwrap()[0].clone();
        let final_statevector_trafo =
            circuit_transformed_result.get("ro_trafo").unwrap()[0].clone();
        assert_eq!(final_statevector.len(), final_statevector_trafo.len());

        let (direct_val, transformed_val) = final_statevector
            .iter()
            .zip(final_statevector_trafo.iter())
            .find(|(x, _)| x.norm() > 1e-6)
            .unwrap();
        let global_phase_factor = transformed_val / direct_val;
        for (direct_val, transformed_val) in
            final_statevector.iter().zip(final_statevector_trafo.iter())
        {
            if (global_phase_factor * direct_val - transformed_val).norm() > 1e-6 {
                println!(
                    "direct val {}  transformed val {} difference norm {}, global phase factor {}",
                    direct_val,
                    transformed_val,
                    (direct_val - transformed_val).norm(),
                    global_phase_factor
                );
                panic!();
            }
        }
    }
}

#[test_case(0_u64; "seed0")]
#[test_case(1_u64; "seed1")]
#[test_case(2_u64; "seed2")]
#[test_case(3_u64; "seed3")]
fn test_random_circuits(seed: u64) {
    let number_qubits = 6;
    let circuit_length = 1000;
    let dimension = 2_usize.pow(number_qubits as u32);
    let mut circuit = construct_random_circuit(circuit_length, number_qubits, seed);
    circuit += MultiQubitZZ::new(vec![0, 1, 2], 0.1.into());
    let mut circuit_trafo =
        circuit.clone() + DefinitionComplex::new("ro_trafo".to_string(), dimension, true);
    circuit += DefinitionComplex::new("ro".to_string(), dimension, true);

    circuit += PragmaGetStateVector::new("ro".to_string(), None);
    circuit_trafo += PragmaGetStateVector::new("ro_trafo".to_string(), None);

    let transformed_circuit = virtual_z_replacement_circuit(&circuit_trafo, None, true)
        .unwrap()
        .0;
    for i in 0..dimension {
        let mut initial_statevector: Array1<Complex64> = Array1::zeros(dimension);
        initial_statevector[i] = Complex64::new(1.0, 0.0);
        let mut tmp_circuit = Circuit::new();
        let mut tmp_transformed_circuit = Circuit::new();

        tmp_circuit += PragmaSetStateVector::new(initial_statevector.clone());
        tmp_transformed_circuit += PragmaSetStateVector::new(initial_statevector.clone());
        tmp_circuit += circuit.clone();
        tmp_transformed_circuit += transformed_circuit.clone();

        let backend = roqoqo_quest::Backend::new(number_qubits, Some(vec![2, 4]));
        let (_, _, circuit_result) = backend.run_circuit(&tmp_circuit).unwrap();
        let (_, _, circuit_transformed_result) =
            backend.run_circuit(&tmp_transformed_circuit).unwrap();
        let final_statevector = circuit_result.get("ro").unwrap()[0].clone();
        let final_statevector_trafo =
            circuit_transformed_result.get("ro_trafo").unwrap()[0].clone();
        assert_eq!(final_statevector.len(), final_statevector_trafo.len());

        let (direct_val, transformed_val) = final_statevector
            .iter()
            .zip(final_statevector_trafo.iter())
            .find(|(x, _)| x.norm() > 1e-6)
            .unwrap();
        let global_phase_factor = transformed_val / direct_val;
        for (direct_val, transformed_val) in
            final_statevector.iter().zip(final_statevector_trafo.iter())
        {
            if (global_phase_factor * direct_val - transformed_val).norm() > 1e-6 {
                println!(
                    "direct val {}  transformed val {} difference norm {}, global phase factor {}",
                    direct_val,
                    transformed_val,
                    (direct_val - transformed_val).norm(),
                    global_phase_factor
                );
                panic!();
            }
        }
    }
}
