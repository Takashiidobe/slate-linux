#![no_std]
#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals, rustdoc::all)]
#![allow(suspicious_runtime_symbol_definitions)]
#![feature(linkage)]
#![feature(link_llvm_intrinsics)]
#![feature(used_with_arg)]

#[path = "tables.rs"]
mod __slate_unit_tables;
#[path = "osi.rs"]
mod __slate_unit_osi;
#[path = "osl.rs"]
mod __slate_unit_osl;
#[path = "utils.rs"]
mod __slate_unit_utils;
#[path = "reboot.rs"]
mod __slate_unit_reboot;
#[path = "nvs.rs"]
mod __slate_unit_nvs;
#[path = "wakeup.rs"]
mod __slate_unit_wakeup;
#[path = "sleep.rs"]
mod __slate_unit_sleep;
#[path = "device_sysfs.rs"]
mod __slate_unit_device_sysfs;
#[path = "device_pm.rs"]
mod __slate_unit_device_pm;
#[path = "proc.rs"]
mod __slate_unit_proc;
#[path = "bus.rs"]
mod __slate_unit_bus;
#[path = "glue.rs"]
mod __slate_unit_glue;
#[path = "scan.rs"]
mod __slate_unit_scan;
#[path = "mipi_disco_img.rs"]
mod __slate_unit_mipi_disco_img;
#[path = "resource.rs"]
mod __slate_unit_resource;
#[path = "acpi_processor.rs"]
mod __slate_unit_acpi_processor;
#[path = "processor_core.rs"]
mod __slate_unit_processor_core;
#[path = "processor_pdc.rs"]
mod __slate_unit_processor_pdc;
#[path = "ec.rs"]
mod __slate_unit_ec;
#[path = "dock.rs"]
mod __slate_unit_dock;
#[path = "pci_root.rs"]
mod __slate_unit_pci_root;
#[path = "pci_link.rs"]
mod __slate_unit_pci_link;
#[path = "pci_irq.rs"]
mod __slate_unit_pci_irq;
#[path = "acpi_apd.rs"]
mod __slate_unit_acpi_apd;
#[path = "acpi_platform.rs"]
mod __slate_unit_acpi_platform;
#[path = "acpi_pnp.rs"]
mod __slate_unit_acpi_pnp;
#[path = "power.rs"]
mod __slate_unit_power;
#[path = "event.rs"]
mod __slate_unit_event;
#[path = "evged.rs"]
mod __slate_unit_evged;
#[path = "sysfs.rs"]
mod __slate_unit_sysfs;
#[path = "property.rs"]
mod __slate_unit_property;
#[path = "debugfs.rs"]
mod __slate_unit_debugfs;
#[path = "acpi_lpat.rs"]
mod __slate_unit_acpi_lpat;
#[path = "acpi_lpit.rs"]
mod __slate_unit_acpi_lpit;
#[path = "prmt.rs"]
mod __slate_unit_prmt;
#[path = "acpi_pcc.rs"]
mod __slate_unit_acpi_pcc;
#[path = "acpi_mrrm.rs"]
mod __slate_unit_acpi_mrrm;
#[path = "ac.rs"]
mod __slate_unit_ac;
#[path = "button.rs"]
mod __slate_unit_button;
#[path = "fan_core.rs"]
mod __slate_unit_fan_core;
#[path = "fan_attr.rs"]
mod __slate_unit_fan_attr;
#[path = "fan_hwmon.rs"]
mod __slate_unit_fan_hwmon;
#[path = "acpi_video.rs"]
mod __slate_unit_acpi_video;
#[path = "video_detect.rs"]
mod __slate_unit_video_detect;
#[path = "processor_driver.rs"]
mod __slate_unit_processor_driver;
#[path = "processor_thermal.rs"]
mod __slate_unit_processor_thermal;
#[path = "processor_idle.rs"]
mod __slate_unit_processor_idle;
#[path = "processor_throttling.rs"]
mod __slate_unit_processor_throttling;
#[path = "processor_perflib.rs"]
mod __slate_unit_processor_perflib;
#[path = "container.rs"]
mod __slate_unit_container;
#[path = "thermal_lib.rs"]
mod __slate_unit_thermal_lib;
#[path = "thermal.rs"]
mod __slate_unit_thermal;
#[path = "platform_profile.rs"]
mod __slate_unit_platform_profile;
#[path = "nhlt.rs"]
mod __slate_unit_nhlt;
#[path = "acpi_memhotplug.rs"]
mod __slate_unit_acpi_memhotplug;
#[path = "ioapic.rs"]
mod __slate_unit_ioapic;
#[path = "battery.rs"]
mod __slate_unit_battery;
#[path = "bgrt.rs"]
mod __slate_unit_bgrt;
#[path = "cppc_acpi.rs"]
mod __slate_unit_cppc_acpi;
#[path = "spcr.rs"]
mod __slate_unit_spcr;

mod __slate_panic {
    #[panic_handler]
    fn __slate_panic_handler(_: &core::panic::PanicInfo) -> ! {
        loop {}
    }

    #[linkage = "weak"]
    #[unsafe(no_mangle)]
    extern "C" fn rust_eh_personality() {}
}

