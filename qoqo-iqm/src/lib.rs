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

//! Python interface of roqoqo-iqm
//!

#![deny(missing_docs)]
#![warn(rustdoc::private_intra_doc_links)]
#![warn(rustdoc::missing_crate_level_docs)]
#![warn(rustdoc::missing_doc_code_examples)]
#![warn(rustdoc::private_doc_tests)]

use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::wrap_pymodule;

/// Qoqo-IQM devices
pub mod devices;
pub use devices::*;

mod backend;
pub use backend::BackendWrapper;

/// IQM python interface
///
/// Provides the devices that are used to execute quantum programs with the IQM backend, as well as the IQM backend.
#[pymodule]
fn qoqo_iqm(_py: Python, module: &Bound<PyModule>) -> PyResult<()> {
    module.add_class::<BackendWrapper>()?;
    module.add_class::<DenebDeviceWrapper>()?;
    module.add_class::<GarnetDeviceWrapper>()?;

    let wrapper = wrap_pymodule!(devices::iqm_devices);
    module.add_wrapped(wrapper)?;

    // Adding nice imports corresponding to maturin example
    let system = PyModule::import(_py, "sys")?;
    let binding = system.getattr("modules")?;
    let system_modules: &Bound<PyDict> = binding.downcast()?;
    system_modules.set_item("qoqo_iqm.devices", module.getattr("iqm_devices")?)?;
    Ok(())
}
