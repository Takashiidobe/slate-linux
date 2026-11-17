#![no_std]
#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals, rustdoc::all)]
#![feature(linkage)]
#![feature(link_llvm_intrinsics)]
#![feature(used_with_arg)]

#[path = "power_supply_core.rs"]
mod __slate_unit_power_supply_core;
#[path = "power_supply_sysfs.rs"]
mod __slate_unit_power_supply_sysfs;
#[path = "power_supply_leds.rs"]
mod __slate_unit_power_supply_leds;
#[path = "power_supply_hwmon.rs"]
mod __slate_unit_power_supply_hwmon;

mod __slate_panic {
    #[panic_handler]
    fn __slate_panic_handler(_: &core::panic::PanicInfo) -> ! {
        loop {}
    }

    #[linkage = "weak"]
    #[unsafe(no_mangle)]
    extern "C" fn rust_eh_personality() {}
}

