#![no_std]
#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals, rustdoc::all)]
#![allow(suspicious_runtime_symbol_definitions)]
#![feature(linkage)]
#![feature(link_llvm_intrinsics)]
#![feature(used_with_arg)]

#[path = "scsi.rs"]
mod __slate_unit_scsi;
#[path = "hosts.rs"]
mod __slate_unit_hosts;
#[path = "scsi_ioctl.rs"]
mod __slate_unit_scsi_ioctl;
#[path = "scsicam.rs"]
mod __slate_unit_scsicam;
#[path = "scsi_error.rs"]
mod __slate_unit_scsi_error;
#[path = "scsi_lib.rs"]
mod __slate_unit_scsi_lib;
#[path = "constants.rs"]
mod __slate_unit_constants;
#[path = "scsi_lib_dma.rs"]
mod __slate_unit_scsi_lib_dma;
#[path = "scsi_scan.rs"]
mod __slate_unit_scsi_scan;
#[path = "scsi_sysfs.rs"]
mod __slate_unit_scsi_sysfs;
#[path = "scsi_devinfo.rs"]
mod __slate_unit_scsi_devinfo;
#[path = "scsi_sysctl.rs"]
mod __slate_unit_scsi_sysctl;
#[path = "scsi_proc.rs"]
mod __slate_unit_scsi_proc;
#[path = "scsi_debugfs.rs"]
mod __slate_unit_scsi_debugfs;
#[path = "scsi_trace.rs"]
mod __slate_unit_scsi_trace;
#[path = "scsi_logging.rs"]
mod __slate_unit_scsi_logging;
#[path = "scsi_pm.rs"]
mod __slate_unit_scsi_pm;
#[path = "scsi_bsg.rs"]
mod __slate_unit_scsi_bsg;
#[path = "scsi_common.rs"]
mod __slate_unit_scsi_common;
#[path = "scsi_transport_spi.rs"]
mod __slate_unit_scsi_transport_spi;
#[path = "virtio_scsi.rs"]
mod __slate_unit_virtio_scsi;
#[path = "sd.rs"]
mod __slate_unit_sd;
#[path = "sr.rs"]
mod __slate_unit_sr;
#[path = "sr_ioctl.rs"]
mod __slate_unit_sr_ioctl;
#[path = "sr_vendor.rs"]
mod __slate_unit_sr_vendor;
#[path = "sg.rs"]
mod __slate_unit_sg;

mod __slate_panic {
    #[panic_handler]
    fn __slate_panic_handler(_: &core::panic::PanicInfo) -> ! {
        loop {}
    }

    #[linkage = "weak"]
    #[unsafe(no_mangle)]
    extern "C" fn rust_eh_personality() {}
}

