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

use pyo3::prelude::*;
use pyo3::Python;
use qoqo_iqm::devices;
use qoqo_iqm::BackendWrapper;
use std::env;

#[test]
fn test_creating_backend_deneb_device() {
    // Initialize python interpreter in a thread-safe way
    pyo3::prepare_freethreaded_python();

    // Test if Backend is created successfully with a dummy access token
    Python::with_gil(|py| {
        // get Python type (i.e. Python class) corresponding to DenebDeviceWrapper Rust type
        let device_type = py.get_type::<devices::DenebDeviceWrapper>();
        let device = device_type
            // Instantiate Python class
            .call0()
            .unwrap();
        let backend_type = py.get_type::<BackendWrapper>();
        let _backend = backend_type
            .call1((
                device.downcast::<devices::DenebDeviceWrapper>().unwrap(),
                "DUMMY_ACCESS_TOKEN",
            ))
            .unwrap()
            .downcast::<BackendWrapper>()
            .unwrap();
    });

    if env::var("IQM_TOKEN").is_ok() {
        // Test if Backend correctly retrieves access token from environment variable
        Python::with_gil(|py| {
            let device_type = py.get_type::<devices::DenebDeviceWrapper>();
            let device = device_type.call0().unwrap();
            let backend_type = py.get_type::<BackendWrapper>();
            let _backend = backend_type
                .call1((device.downcast::<devices::DenebDeviceWrapper>().unwrap(),))
                .unwrap()
                .downcast::<BackendWrapper>()
                .unwrap();
        })
    } else {
        // If the environment variable IQM_TOKEN is not set and an access token is not provided,
        // creation of the Backend should fail
        Python::with_gil(|py| {
            let device_type = py.get_type::<devices::DenebDeviceWrapper>();
            let device = device_type.call0().unwrap();
            let backend_type = py.get_type::<BackendWrapper>();
            let backend =
                backend_type.call1((device.downcast::<devices::DenebDeviceWrapper>().unwrap(),));
            match backend {
                Err(_) => (),
                _ => panic!("Missing Access Token does not return correct error"),
            }
        })
    }
}

#[test]
fn test_creating_backend_resonator_free_device() {
    // Initialize python interpreter in a thread-safe way
    pyo3::prepare_freethreaded_python();

    // Test if Backend is created successfully with a dummy access token
    Python::with_gil(|py| {
        // get Python type (i.e. Python class) corresponding to ResonatorFreeDeviceWrapper Rust type
        let device_type = py.get_type::<devices::ResonatorFreeDeviceWrapper>();
        let device = device_type
            // Instantiate Python class
            .call0()
            .unwrap();
        let backend_type = py.get_type::<BackendWrapper>();
        let _backend = backend_type
            .call1((
                device
                    .downcast::<devices::ResonatorFreeDeviceWrapper>()
                    .unwrap(),
                "DUMMY_ACCESS_TOKEN",
            ))
            .unwrap()
            .downcast::<BackendWrapper>()
            .unwrap();
    });

    if env::var("IQM_TOKEN").is_ok() {
        // Test if Backend correctly retrieves access token from environment variable
        Python::with_gil(|py| {
            let device_type = py.get_type::<devices::ResonatorFreeDeviceWrapper>();
            let device = device_type.call0().unwrap();
            let backend_type = py.get_type::<BackendWrapper>();
            let _backend = backend_type
                .call1((device
                    .downcast::<devices::ResonatorFreeDeviceWrapper>()
                    .unwrap(),))
                .unwrap()
                .downcast::<BackendWrapper>()
                .unwrap();
        })
    } else {
        // If the environment variable IQM_TOKEN is not set and an access token is not provided,
        // creation of the Backend should fail
        Python::with_gil(|py| {
            let device_type = py.get_type::<devices::ResonatorFreeDeviceWrapper>();
            let device = device_type.call0().unwrap();
            let backend_type = py.get_type::<BackendWrapper>();
            let backend = backend_type.call1((device
                .downcast::<devices::ResonatorFreeDeviceWrapper>()
                .unwrap(),));
            match backend {
                Err(_) => (),
                _ => panic!("Missing Access Token does not return correct error"),
            }
        })
    }
}

#[test]
fn test_creating_backend_garnet_device() {
    // Initialize python interpreter in a thread-safe way
    pyo3::prepare_freethreaded_python();

    // Test if Backend is created successfully with a dummy access token
    Python::with_gil(|py| {
        let device_type = py.get_type::<devices::GarnetDeviceWrapper>();
        let device = device_type
            // Instantiate Python class
            .call0()
            .unwrap();
        let backend_type = py.get_type::<BackendWrapper>();
        let _backend = backend_type
            .call1((
                device.downcast::<devices::GarnetDeviceWrapper>().unwrap(),
                "DUMMY_ACCESS_TOKEN",
            ))
            .unwrap()
            .downcast::<BackendWrapper>()
            .unwrap();
    });

    if env::var("IQM_TOKEN").is_ok() {
        // Test if Backend correctly retrieves access token from environment variable
        Python::with_gil(|py| {
            let device_type = py.get_type::<devices::GarnetDeviceWrapper>();
            let device = device_type.call0().unwrap();
            let backend_type = py.get_type::<BackendWrapper>();
            let _backend = backend_type
                .call1((device.downcast::<devices::GarnetDeviceWrapper>().unwrap(),))
                .unwrap()
                .downcast::<BackendWrapper>()
                .unwrap();
        })
    } else {
        // If the environment variable IQM_TOKEN is not set and an access token is not provided, creation of the Backend should fail
        Python::with_gil(|py| {
            let device_type = py.get_type::<devices::GarnetDeviceWrapper>();
            let device = device_type.call0().unwrap();
            let backend_type = py.get_type::<BackendWrapper>();
            let backend =
                backend_type.call1((device.downcast::<devices::GarnetDeviceWrapper>().unwrap(),));
            match backend {
                Err(_) => (),
                _ => panic!("Missing Access Token does not return correct error"),
            }
        })
    }
}
