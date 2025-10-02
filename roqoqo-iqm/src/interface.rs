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

use itertools::Itertools;
use roqoqo::Circuit;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::f64::consts::PI;

use qoqo_calculator::CalculatorFloat;
use roqoqo::operations::*;
use roqoqo::RoqoqoBackendError;

use crate::IqmBackendError;

// HashMap that associates to each register name the indices in the register that are being affected
// by measurements, and the length of the register. This information is needed to post process the
// results returned by the server.
pub(crate) type MeasuredQubitsMap = HashMap<String, (Vec<usize>, usize)>;

// Pragma operations that are ignored by backend and do not throw an error
const ALLOWED_OPERATIONS: &[&str; 8] = &[
    "PragmaBoostNoise",
    "PragmaStopParallelBlock",
    "PragmaGlobalPhase",
    "InputSymbolic",
    "InputBit",
    "PragmaRepeatedMeasurement",
    "PragmaStartDecompositionBlock",
    "PragmaStopDecompositionBlock",
];

/// Representation for quantum circuits accepted by the IQM REST API.
///
/// Roqoqo does not have a `name` identifier for quantum circuits, but it is needed when
/// submitting to the IQM backend.
#[derive(PartialEq, Debug, Clone, Serialize, Deserialize)]
pub struct IqmCircuit {
    /// Name of the circuit
    pub name: String,
    /// Vector of instructions accepted by the IQM REST API
    pub instructions: Vec<IqmInstruction>,
    /// Optional arbitrary metadata associated with the circuit. Here used to store the lists of
    /// measured qubits for each register, used for processing the results.
    pub metadata: Option<MeasuredQubitsMap>,
}

/// Representation for instructions accepted by the IQM REST API
#[derive(PartialEq, Debug, Clone, Serialize, Deserialize)]
pub struct IqmInstruction {
    /// Identifies the type of instruction, which can be a gate, a measurement or a barrier
    pub name: String,
    /// The qubits involved in the operation
    pub qubits: Vec<String>,
    /// Arguments of the instruction. They depend on the type of operation, and can hold both gate
    /// parameters or measurement names. The latter are used as register names when converting the
    /// results into roqoqo registers.
    pub args: HashMap<String, CalculatorFloat>,
}

/// Converts all operations in a [roqoqo::Circuit] into instructions for IQM Hardware.
///
/// # Arguments
///
/// * `circuit` - The [roqoqo::Circuit] that is converted
/// * `device_number_qubits` - The number of qubits of the backend device. It is used to know how
///   many qubits to measure with [roqoqo::operations::PragmaRepeatedMeasurement]
/// * `number_measurements_internal` - If set, the number of measurements that has been overwritten
///   in the backend
/// * `circuit_index` - Index of the circuit in the batch, needed to assign a unique name to the circuit.
///
/// # Returns
///
/// * `Ok(IqmCircuit, usize)` - Converted circuit and number of measurements
/// * `Err(RoqoqoBackendError::OperationNotInBackend)` - Error when [roqoqo::operations::Operation]
///   can not be converted
pub fn call_circuit<'a>(
    circuit: impl Iterator<Item = &'a Operation>,
    device_number_qubits: usize,
    number_measurements_internal: Option<usize>,
    circuit_index: usize,
) -> Result<(IqmCircuit, usize), IqmBackendError> {
    let mut circuit_vec: Vec<IqmInstruction> = Vec::new();
    let mut number_measurements: usize = 1;
    let mut measured_qubits: Vec<usize> = vec![];
    let mut measured_qubits_map: MeasuredQubitsMap = HashMap::new();

    for op in circuit {
        match op {
            Operation::DefinitionBit(o) => {
                let name = (*o).name().to_string();
                if *o.is_output() {
                    measured_qubits_map.insert(name, (vec![], *o.length()));
                }
            }
            Operation::MeasureQubit(o) => {
                let readout = o.readout().clone();
                measured_qubits.push(*o.qubit());

                match measured_qubits_map.get_mut(&readout) {
                    Some(x) => x.0.push(*o.readout_index()),
                    None => {
                        return Err(IqmBackendError::InvalidCircuit {
                            msg: "A MeasureQubit operation is writing to an undefined register."
                                .to_string(),
                        })
                    }
                }
                let mut found: bool = false;
                // Check if we already have a measurement to the same register
                // if yes, add the qubit being measured to that measurement
                for instr in circuit_vec.iter_mut().filter(|x| x.name == "measure") {
                    let meas_readout = instr.args.get("key").expect(
                        "An IqmInstruction measurement must contain a `key` entry in \
                         the `args` field.",
                    );
                    if let CalculatorFloat::Str(s) = meas_readout {
                        if s == &readout {
                            found = true;
                            let iqm_qubit = _convert_qubit_name_qoqo_to_iqm(*o.qubit());
                            if !instr.qubits.contains(&iqm_qubit) {
                                instr.qubits.push(iqm_qubit);
                            } else {
                                return Err(IqmBackendError::InvalidCircuit {
                                    msg: format!("Qubit {} is being measured twice.", *o.qubit()),
                                });
                            }
                            break;
                        }
                    }
                }
                if !found {
                    // If no measurement to the same register was found, create a new IqmInstruction
                    let meas = IqmInstruction {
                        name: "measure".to_string(),
                        qubits: vec![_convert_qubit_name_qoqo_to_iqm(*o.qubit())],
                        args: HashMap::from([("key".to_string(), readout.into())]),
                    };
                    circuit_vec.push(meas)
                }
            }
            Operation::PragmaSetNumberOfMeasurements(o) => {
                if number_measurements > 1 && number_measurements != *o.number_measurements() {
                    return Err(IqmBackendError::InvalidCircuit {
                        msg: "All PragmaSetNumberOfMeasurements in the circuit must have the same /
                              number of measurements."
                            .to_string(),
                    });
                }
                number_measurements = *o.number_measurements();

                let readout = o.readout().clone();
                let readout_register = measured_qubits_map.get(&readout);

                match readout_register {
                    None => {
                        return Err(IqmBackendError::InvalidCircuit {
                            msg: format!(
                                "PragmaSetNumberOfMeasurements writes to an undefined register {}",
                                &readout
                            ),
                        })
                    }
                    Some(reg) => {
                        let readout_length = reg.1;
                        if measured_qubits.len() > readout_length {
                            return Err(IqmBackendError::RegisterTooSmall { name: readout });
                        }

                        // remove MeasureQubit operations
                        let mut old_measurement_indices = vec![];
                        for (i, meas) in circuit_vec.iter().enumerate() {
                            if meas.name == "measure" {
                                old_measurement_indices.push(i);
                            }
                        }
                        // iterate indices in reverse order to avoid shifting the entries of circuit_vec
                        for i in old_measurement_indices.into_iter().rev() {
                            circuit_vec.remove(i);
                        }

                        // update register mapping with the only register specified by PragmaSetNumberOfMeasurements
                        measured_qubits_map = HashMap::new();
                        measured_qubits_map
                            .insert(readout.clone(), (measured_qubits.clone(), readout_length));

                        // add single measurement instruction for all the qubits that were measured with MeasureQubit
                        let meas = IqmInstruction {
                            name: "measure".to_string(),
                            qubits: measured_qubits
                                .iter()
                                .map(|x| _convert_qubit_name_qoqo_to_iqm(*x))
                                .collect(),
                            args: HashMap::from([(
                                "key".to_string(),
                                CalculatorFloat::Str(readout),
                            )]),
                        };
                        circuit_vec.push(meas)
                    }
                }
            }
            Operation::PragmaRepeatedMeasurement(o) => {
                if number_measurements > 1 {
                    return Err(IqmBackendError::InvalidCircuit {
                        msg: "Only one repeated measurement is allowed in the circuit.".to_string(),
                    });
                }
                if !measured_qubits.is_empty() {
                    return Err(IqmBackendError::InvalidCircuit {
                        msg: "Some qubits are being measured twice.".to_string(),
                    });
                }

                number_measurements = *o.number_measurements();
                let readout = o.readout().clone();

                match o.qubit_mapping() {
                    None => match measured_qubits_map.get(&readout) {
                        None => {
                            return Err(IqmBackendError::InvalidCircuit {
                                msg: "A PragmaRepeatedMeasurement operation is writing to an \
                                      undefined register."
                                    .to_string(),
                            })
                        }
                        Some(reg) => {
                            let readout_length = reg.1;
                            let readout_name = o.readout().to_string();
                            measured_qubits_map.insert(
                                readout_name,
                                ((0..readout_length).collect(), readout_length),
                            );
                        }
                    },
                    Some(map) => match measured_qubits_map.get_mut(o.readout()) {
                        Some(x) => {
                            for qubit in map.keys().sorted() {
                                x.0.push(map[qubit])
                            }
                        }
                        None => {
                            return Err(IqmBackendError::InvalidCircuit {
                                msg: "A PragmaRepeatedMeasurement operation is writing to an \
                                     undefined register."
                                    .to_string(),
                            })
                        }
                    },
                }

                let measure_all = IqmInstruction {
                    name: "measure".to_string(),
                    qubits: _convert_all_qubit_names(device_number_qubits),
                    args: HashMap::from([("key".to_string(), CalculatorFloat::Str(readout))]),
                };
                circuit_vec.push(measure_all);
            }
            Operation::PragmaLoop(o) => {
                let reps_ref =
                    o.repetitions()
                        .float()
                        .map_err(|_| IqmBackendError::InvalidCircuit {
                            msg: "Only Loops with non-symbolic repetitions are supported by the \
                                  backend."
                                .to_string(),
                        })?;
                let reps = (*reps_ref) as i32;

                for _ in 0..reps {
                    for i in o.circuit().iter() {
                        if let Some(instruction) = call_operation(i)? {
                            circuit_vec.push(instruction);
                        }
                    }
                }
            }
            _ => {
                if let Some(instruction) = call_operation(op)? {
                    circuit_vec.push(instruction)
                }
            }
        };
    }

    if let Some(n) = number_measurements_internal {
        number_measurements = n
    }

    let iqm_circuit = IqmCircuit {
        name: format!("qc_{}", circuit_index),
        instructions: circuit_vec,
        metadata: Some(measured_qubits_map),
    };

    Ok((iqm_circuit, number_measurements))
}

/// Converts a [roqoqo::operations::Operation] into a native instruction for IQM Hardware
///
/// # Arguments
///
/// * `operation` - The [roqoqo::operations::Operation] that is converted
///
/// # Returns
///
/// * `Ok(IqmInstruction)` - Converted instruction  
/// * `Err(RoqoqoBackendError::OperationNotInBackend)` - Error when [roqoqo::operations::Operation] can not be converted
pub fn call_operation(operation: &Operation) -> Result<Option<IqmInstruction>, RoqoqoBackendError> {
    let mut op_parameters = HashMap::new();

    match operation {
        Operation::RotateXY(op) => {
            // Angles are measured in units of 2*PI in the IQM API
            op_parameters.insert(
                "angle_t".to_string(),
                CalculatorFloat::Float(*op.theta().float()? / (2.0 * PI)),
            );
            op_parameters.insert(
                "phase_t".to_string(),
                CalculatorFloat::Float(*op.phi().float()? / (2.0 * PI)),
            );

            Ok(Some(IqmInstruction {
                name: "prx".to_string(),
                qubits: vec![_convert_qubit_name_qoqo_to_iqm(*op.qubit())],
                args: op_parameters,
            }))
        }
        Operation::ControlledPauliZ(op) => {
            let control = _convert_qubit_name_qoqo_to_iqm(*op.control());
            let target = _convert_qubit_name_qoqo_to_iqm(*op.target());

            Ok(Some(IqmInstruction {
                name: "cz".to_string(),
                qubits: vec![control, target],
                args: op_parameters,
            }))
        }
        Operation::CZQubitResonator(op) => {
            let control = _convert_qubit_name_qoqo_to_iqm(*op.qubit());
            let resonator = _convert_resonator_name_qoqo_to_iqm(*op.mode());

            Ok(Some(IqmInstruction {
                name: "cz".to_string(),
                qubits: vec![control, resonator],
                args: op_parameters,
            }))
        }
        Operation::SingleExcitationLoad(op) => {
            let control = _convert_qubit_name_qoqo_to_iqm(*op.qubit());
            let resonator = _convert_resonator_name_qoqo_to_iqm(*op.mode());

            Ok(Some(IqmInstruction {
                name: "move".to_string(),
                qubits: vec![control, resonator],
                args: op_parameters,
            }))
        }
        Operation::SingleExcitationStore(op) => {
            let control = _convert_qubit_name_qoqo_to_iqm(*op.qubit());
            let resonator = _convert_resonator_name_qoqo_to_iqm(*op.mode());

            Ok(Some(IqmInstruction {
                name: "move".to_string(),
                qubits: vec![control, resonator],
                args: op_parameters,
            }))
        }
        _ => {
            if ALLOWED_OPERATIONS.contains(&operation.hqslang()) {
                Ok(None)
            } else {
                Err(RoqoqoBackendError::OperationNotInBackend {
                    backend: "IQM",
                    hqslang: operation.hqslang(),
                })
            }
        }
    }
}

/// Convert a qubit number into the format accepted by IQM.
// e.g. "QB2" for qoqo_qubit number 1 (IQM qubits start from 1)
#[inline]
fn _convert_qubit_name_qoqo_to_iqm(qoqo_qubit: usize) -> String {
    format!("QB{}", qoqo_qubit + 1)
}

/// Convert a resonator number into the format accepted by IQM.
#[inline]
fn _convert_resonator_name_qoqo_to_iqm(_resonator_index: usize) -> String {
    "COMP_R".to_string()
}

/// Create a vector will all qubit names, in the format accepted by IQM
#[inline]
fn _convert_all_qubit_names(number_qubits: usize) -> Vec<String> {
    let mut qubit_vec = vec![];
    for i in 1..=number_qubits {
        qubit_vec.push(format!("QB{}", i))
    }
    qubit_vec
}

/// Applies virtual Z gate optimization.
///
/// On platforms where the RotateXY gate is available and the
/// natural two-qubit gate is diagonal in the Z-basis,
/// RotateZ gates can be replaced by virtual changes (rotations)
/// of the qubit basis in the XY-Plane. This is also referred to as
/// VirtualRotateZ gates optimization.
///
/// # Arguments
///
/// * `input` - The Circuit that is optimized.
///
/// # Returns
///
/// * `Ok(Circuit)` - Converted circuit when input was a Circuit.
/// * `Err(IqmBackendError)` - Optimization failed.
pub fn virtual_z_replacement_circuit(
    circuit: &Circuit,
    rotation_map: Option<HashMap<usize, CalculatorFloat>>,
    apply_final_rotz: bool,
) -> Result<(Circuit, Option<HashMap<usize, CalculatorFloat>>), IqmBackendError> {
    let mut new_circuit = Circuit::new();
    let mut rotation_map = rotation_map.unwrap_or_default();

    for op in circuit.iter() {
        if let Ok(two_qubit_op) = TwoQubitGateOperation::try_from(op) {
            match two_qubit_op {
                TwoQubitGateOperation::ControlledPhaseShift(_) => new_circuit += two_qubit_op,
                TwoQubitGateOperation::ControlledPauliZ(_) => new_circuit += two_qubit_op,
                TwoQubitGateOperation::PhaseShiftedControlledZ(_) => new_circuit += two_qubit_op,
                TwoQubitGateOperation::PhaseShiftedControlledPhase(_) => {
                    new_circuit += two_qubit_op
                }
                _ => {
                    return Err(IqmBackendError::InvalidCircuit {
                        msg: format!(
                            "condition: all two qubit gates are diagonal when virtualZ replacement is used.\ngate: {}",
                            two_qubit_op.hqslang(),
                        ),
                    })
                }
            }
        } else if let Ok(single_qubit_op) = SingleQubitGateOperation::try_from(op) {
            match single_qubit_op {
                SingleQubitGateOperation::RotateZ(inner_op) => {
                    update_phase_map(&mut rotation_map, inner_op.qubit(), inner_op.theta())
                }
                SingleQubitGateOperation::RotateX(inner_op) => {
                    add_rotxy(
                        &rotation_map,
                        inner_op.qubit(),
                        &mut new_circuit,
                        inner_op.theta().clone(),
                        CalculatorFloat::ZERO,
                    );
                }
                SingleQubitGateOperation::RotateY(inner_op) => {
                    add_rotxy(
                        &rotation_map,
                        inner_op.qubit(),
                        &mut new_circuit,
                        inner_op.theta().clone(),
                        CalculatorFloat::FRAC_PI_2,
                    );
                }
                SingleQubitGateOperation::PauliX(inner_op) => {
                    add_rotxy(
                        &rotation_map,
                        inner_op.qubit(),
                        &mut new_circuit,
                        CalculatorFloat::PI,
                        CalculatorFloat::ZERO,
                    );
                }
                SingleQubitGateOperation::PauliY(inner_op) => {
                    add_rotxy(
                        &rotation_map,
                        inner_op.qubit(),
                        &mut new_circuit,
                        CalculatorFloat::PI,
                        CalculatorFloat::FRAC_PI_2,
                    );
                }
                SingleQubitGateOperation::PauliZ(inner_op) => {
                    update_phase_map(&mut rotation_map, inner_op.qubit(), &CalculatorFloat::PI)
                }
                SingleQubitGateOperation::SqrtPauliX(inner_op) => {
                    add_rotxy(
                        &rotation_map,
                        inner_op.qubit(),
                        &mut new_circuit,
                        CalculatorFloat::FRAC_PI_2,
                        CalculatorFloat::ZERO,
                    );
                }
                SingleQubitGateOperation::InvSqrtPauliX(inner_op) => {
                    add_rotxy(
                        &rotation_map,
                        inner_op.qubit(),
                        &mut new_circuit,
                        -CalculatorFloat::FRAC_PI_2,
                        CalculatorFloat::ZERO,
                    );
                }
                SingleQubitGateOperation::PhaseShiftState1(inner_op) => {
                    update_phase_map(&mut rotation_map, inner_op.qubit(), inner_op.theta())
                }
                SingleQubitGateOperation::PhaseShiftState0(inner_op) => update_phase_map(
                    &mut rotation_map,
                    inner_op.qubit(),
                    &(inner_op.theta() * -1),
                ),
                SingleQubitGateOperation::RotateXY(inner_op) => {
                    add_rotxy(
                        &rotation_map,
                        inner_op.qubit(),
                        &mut new_circuit,
                        inner_op.theta().clone(),
                        inner_op.phi().clone(),
                    );
                }
                _ => return Err(IqmBackendError::InvalidCircuit {
                    msg: format!(
                        "all single qubit gates are special cases or RotateZ or RotateXY when virtualZ replacement is used.\ngate: {}",
                        single_qubit_op.hqslang(),
                    ),
                }),
            }
        } else if let Ok(multi_qubit_op) = MultiQubitGateOperation::try_from(op) {
            match multi_qubit_op {
                MultiQubitGateOperation::MultiQubitZZ(_) => {
                    new_circuit += multi_qubit_op;
                }
                _ => {
                    return Err(IqmBackendError::InvalidCircuit { msg:
                            format!(
                            "condition: all multi qubit gates are diagonal when virtualZ replacement is used.\ngate: {}",
                            multi_qubit_op.hqslang(),
                        ),
                    })
                }
            }
        } else {
            match op {
                Operation::MeasureQubit(inner_op) => {
                    // Resetting phase change, because Measurement in Z direction absorbs Z-phase
                    if let Some(phase_value) = rotation_map.get_mut(inner_op.qubit()) {
                        *phase_value = 0.0.into();
                    }
                    new_circuit += inner_op.clone();
                }
                Operation::PragmaLoop(inner_op) => {
                    match inner_op.repetitions() {
                        CalculatorFloat::Float(n) => {
                            // For cases where the number of floats is fixed, the loop can be unrolled and
                            // the full VirtualZ optimization can be applied
                            let n = n.floor() as usize;
                            for _ in 0..n {
                                let (new_inner_circuit, r_map) = virtual_z_replacement_circuit(
                                    inner_op.circuit(),
                                    Some(rotation_map),
                                    false,
                                )?;
                                rotation_map = r_map.expect("Qonvert bug: Unexpectedly did not obtain rotation map from virtual rotate z gate optimisation.");
                                new_circuit += new_inner_circuit;
                            }
                        }
                        CalculatorFloat::Str(_) => {
                            // Undo all basis changes before entering loop. Because it is executed several times
                            // It needs to start in normal basis
                            for (key, val) in rotation_map.iter() {
                                new_circuit += RotateZ::new(*key, -(val.clone()));
                            }
                            rotation_map = HashMap::new();
                            let (new_inner_circuit, _) =
                                virtual_z_replacement_circuit(inner_op.circuit(), None, true)?;
                            new_circuit +=
                                PragmaLoop::new(inner_op.repetitions().clone(), new_inner_circuit);
                        }
                    }
                }
                Operation::PragmaActiveReset(inner_op) => {
                    // Resetting phase change, because Measurement in Z direction absorbs Z-phase
                    if let Some(phase_value) = rotation_map.get_mut(inner_op.qubit()) {
                        *phase_value = 0.0.into();
                    }
                    new_circuit += inner_op.clone();
                }
                Operation::PragmaConditional(inner_op) => {
                    // Undo all basis changes before entering loop. Because it is executed several times
                    // It needs to start in normal basis
                    for (key, val) in rotation_map.iter() {
                        new_circuit += RotateZ::new(*key, -(val.clone()));
                    }
                    rotation_map = HashMap::new();
                    let (new_inner_circuit, _) =
                        virtual_z_replacement_circuit(inner_op.circuit(), None, true)?;
                    new_circuit += PragmaConditional::new(
                        inner_op.condition_register().clone(),
                        *inner_op.condition_index(),
                        new_inner_circuit,
                    );
                }
                Operation::PragmaRepeatedMeasurement(inner_op) => {
                    // Resetting phase change, because Measurement in Z direction absorbs Z-phase
                    rotation_map = HashMap::new();
                    new_circuit += inner_op.clone();
                }
                Operation::PragmaGetStateVector(inner_op) => {
                    for (key, val) in rotation_map.iter() {
                        new_circuit += RotateZ::new(*key, -(val.clone()));
                    }
                    rotation_map = HashMap::new();
                    new_circuit += inner_op.clone();
                }
                Operation::PragmaGetDensityMatrix(inner_op) => {
                    for (key, val) in rotation_map.iter() {
                        new_circuit += RotateZ::new(*key, -(val.clone()));
                    }
                    rotation_map = HashMap::new();
                    new_circuit += inner_op.clone();
                }
                _ => {
                    new_circuit += op.clone();
                }
            }
        }
    }

    if apply_final_rotz {
        for (key, val) in rotation_map.iter() {
            new_circuit += RotateZ::new(*key, -(val.clone()));
        }
    }

    let final_hash_map = if apply_final_rotz {
        None
    } else {
        Some(rotation_map)
    };
    Ok((new_circuit, final_hash_map))
}

#[inline]
fn add_rotxy(
    rotation_map: &HashMap<usize, CalculatorFloat>,
    qubit: &usize,
    new_circuit: &mut Circuit,
    theta: CalculatorFloat,
    offset_rotation: CalculatorFloat,
) {
    let additional_rotation = rotation_map.get(qubit).unwrap_or(&CalculatorFloat::ZERO);
    *new_circuit += RotateXY::new(*qubit, theta, offset_rotation + additional_rotation);
}

#[inline]
fn update_phase_map(
    map: &mut HashMap<usize, CalculatorFloat>,
    qubit: &usize,
    val: &CalculatorFloat,
) {
    match map.get_mut(qubit) {
        Some(v) => *v = v.clone() - val.clone(),
        None => {
            map.insert(*qubit, -val.clone());
        }
    }
}
