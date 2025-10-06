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

//! roqoqo-iqm
//!

#![deny(missing_docs)]
#![warn(rustdoc::private_intra_doc_links)]
#![warn(rustdoc::missing_crate_level_docs)]
#![warn(rustdoc::missing_doc_code_examples)]
#![warn(rustdoc::private_doc_tests)]
#![deny(missing_debug_implementations)]

use roqoqo::RoqoqoBackendError;
use thiserror::Error;

/// Errors that can occur in roqoqo-iqm
#[derive(Error, Debug)]
pub enum IqmBackendError {
    /// Status of submitted job is FAILED
    #[error("Job failed with job ID: {id}\nMessage: {msg}")]
    JobFailed {
        /// Job ID
        id: String,
        /// Message
        msg: String,
    },
    /// Status of submitted job is ABORTED
    #[error("Job with job ID {id} is aborted.")]
    JobAborted {
        /// Job ID
        id: String,
    },
    /// Abortion of a job has failed
    #[error("Could not abort job with ID {id}: {msg}")]
    JobAbortionFailed {
        /// Job ID
        id: String,
        /// Abort response from the endpoint
        msg: String,
    },
    /// Result returned by IQM is empty
    #[error("IQM has returned an empty result for job with ID {id}.")]
    EmptyResult {
        /// Job ID
        id: String,
    },
    /// Circuit passed to the backend is empty
    #[error("An empty circuit was passed to the backend.")]
    EmptyCircuit,
    /// Readout register is too small for the number of qubits in the circuit.
    #[error("Readout register {name} is not large enough for the number of qubits.")]
    RegisterTooSmall {
        /// Name of the readout register
        name: String,
    },
    /// Circuit passed to the backend is invalid
    #[error("{msg}")]
    InvalidCircuit {
        /// Message
        msg: String,
    },
    #[error("{msg}")]
    /// Problem with circuit metadata in the results
    MetadataError {
        /// Message
        msg: String,
    },
    #[error("{msg}")]
    /// Received invalid results from the server
    InvalidResults {
        /// Message
        msg: String,
    },
    /// Transparent propagation of RoqoqoBackendError
    #[error(transparent)]
    RoqoqoBackendError(#[from] RoqoqoBackendError),
}

mod interface;
pub use interface::{
    call_circuit, call_operation, virtual_z_replacement_circuit, IqmCircuit, IqmInstruction,
};

mod backend;
pub use backend::*;

pub mod devices;
pub use devices::*;
