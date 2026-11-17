#![no_std]
#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals, rustdoc::all)]
#![allow(suspicious_runtime_symbol_definitions, deref_nullptr)]
#![feature(linkage)]
#![feature(link_llvm_intrinsics)]
#![feature(used_with_arg)]

#[path = "access.rs"]
mod __slate_unit_access;
#[path = "bus.rs"]
mod __slate_unit_bus;
#[path = "probe.rs"]
mod __slate_unit_probe;
#[path = "host_bridge.rs"]
mod __slate_unit_host_bridge;
#[path = "remove.rs"]
mod __slate_unit_remove;
#[path = "pci.rs"]
mod __slate_unit_pci;
#[path = "pci_driver.rs"]
mod __slate_unit_pci_driver;
#[path = "search.rs"]
mod __slate_unit_search;
#[path = "rebar.rs"]
mod __slate_unit_rebar;
#[path = "rom.rs"]
mod __slate_unit_rom;
#[path = "setup_res.rs"]
mod __slate_unit_setup_res;
#[path = "irq.rs"]
mod __slate_unit_irq;
#[path = "vpd.rs"]
mod __slate_unit_vpd;
#[path = "setup_bus.rs"]
mod __slate_unit_setup_bus;
#[path = "vc.rs"]
mod __slate_unit_vc;
#[path = "mmap.rs"]
mod __slate_unit_mmap;
#[path = "devres.rs"]
mod __slate_unit_devres;
#[path = "pcidev_msi.rs"]
mod __slate_unit_pcidev_msi;
#[path = "api.rs"]
mod __slate_unit_api;
#[path = "msi.rs"]
mod __slate_unit_msi;
#[path = "irqdomain.rs"]
mod __slate_unit_irqdomain;
#[path = "portdrv.rs"]
mod __slate_unit_portdrv;
#[path = "rcec.rs"]
mod __slate_unit_rcec;
#[path = "bwctrl.rs"]
mod __slate_unit_bwctrl;
#[path = "aspm.rs"]
mod __slate_unit_aspm;
#[path = "pme.rs"]
mod __slate_unit_pme;
#[path = "proc.rs"]
mod __slate_unit_proc;
#[path = "pci_sysfs.rs"]
mod __slate_unit_pci_sysfs;
#[path = "slot.rs"]
mod __slate_unit_slot;
#[path = "pci_acpi.rs"]
mod __slate_unit_pci_acpi;
#[path = "iomap.rs"]
mod __slate_unit_iomap;
#[path = "quirks.rs"]
mod __slate_unit_quirks;
#[path = "pci_hotplug_core.rs"]
mod __slate_unit_pci_hotplug_core;
#[path = "acpi_pcihp.rs"]
mod __slate_unit_acpi_pcihp;
#[path = "ats.rs"]
mod __slate_unit_ats;
#[path = "pci_label.rs"]
mod __slate_unit_pci_label;
#[path = "vgaarb.rs"]
mod __slate_unit_vgaarb;
#[path = "setup_cardbus.rs"]
mod __slate_unit_setup_cardbus;
#[path = "trace.rs"]
mod __slate_unit_trace;

mod __slate_panic {
    unsafe extern "C" {
        fn panic(fmt: *const core::ffi::c_char, ...) -> !;
    }

    #[panic_handler]
    fn __slate_panic_handler(_: &core::panic::PanicInfo) -> ! {
        unsafe { panic(c"rust panic\n".as_ptr()) }
    }

    #[linkage = "weak"]
    #[unsafe(no_mangle)]
    extern "C" fn rust_eh_personality() {}
}

