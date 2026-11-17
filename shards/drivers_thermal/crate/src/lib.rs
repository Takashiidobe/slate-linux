#![no_std]
#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals, rustdoc::all)]
#![allow(suspicious_runtime_symbol_definitions)]
#![feature(linkage)]
#![feature(link_llvm_intrinsics)]
#![feature(used_with_arg)]

#[path = "thermal_core.rs"]
mod __slate_unit_thermal_core;
#[path = "thermal_sysfs.rs"]
mod __slate_unit_thermal_sysfs;
#[path = "thermal_trip.rs"]
mod __slate_unit_thermal_trip;
#[path = "thermal_helpers.rs"]
mod __slate_unit_thermal_helpers;
#[path = "thermal_thresholds.rs"]
mod __slate_unit_thermal_thresholds;
#[path = "thermal_netlink.rs"]
mod __slate_unit_thermal_netlink;
#[path = "thermal_hwmon.rs"]
mod __slate_unit_thermal_hwmon;
#[path = "gov_step_wise.rs"]
mod __slate_unit_gov_step_wise;
#[path = "intel_tcc.rs"]
mod __slate_unit_intel_tcc;
#[path = "therm_throt.rs"]
mod __slate_unit_therm_throt;

mod __slate_panic {
    #[panic_handler]
    fn __slate_panic_handler(_: &core::panic::PanicInfo) -> ! {
        loop {}
    }

    #[linkage = "weak"]
    #[unsafe(no_mangle)]
    extern "C" fn rust_eh_personality() {}
}

