#![no_std]
#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals, rustdoc::all)]
#![allow(suspicious_runtime_symbol_definitions)]
#![feature(linkage)]
#![feature(link_llvm_intrinsics)]
#![feature(used_with_arg)]

#[path = "bdev.rs"]
mod __slate_unit_bdev;
#[path = "fops.rs"]
mod __slate_unit_fops;
#[path = "bio.rs"]
mod __slate_unit_bio;
#[path = "elevator.rs"]
mod __slate_unit_elevator;
#[path = "blk_core.rs"]
mod __slate_unit_blk_core;
#[path = "blk_sysfs.rs"]
mod __slate_unit_blk_sysfs;
#[path = "blk_flush.rs"]
mod __slate_unit_blk_flush;
#[path = "blk_settings.rs"]
mod __slate_unit_blk_settings;
#[path = "blk_ioc.rs"]
mod __slate_unit_blk_ioc;
#[path = "blk_map.rs"]
mod __slate_unit_blk_map;
#[path = "blk_merge.rs"]
mod __slate_unit_blk_merge;
#[path = "blk_timeout.rs"]
mod __slate_unit_blk_timeout;
#[path = "blk_lib.rs"]
mod __slate_unit_blk_lib;
#[path = "blk_mq.rs"]
mod __slate_unit_blk_mq;
#[path = "blk_mq_tag.rs"]
mod __slate_unit_blk_mq_tag;
#[path = "blk_mq_dma.rs"]
mod __slate_unit_blk_mq_dma;
#[path = "blk_stat.rs"]
mod __slate_unit_blk_stat;
#[path = "blk_mq_sysfs.rs"]
mod __slate_unit_blk_mq_sysfs;
#[path = "blk_mq_cpumap.rs"]
mod __slate_unit_blk_mq_cpumap;
#[path = "blk_mq_sched.rs"]
mod __slate_unit_blk_mq_sched;
#[path = "ioctl.rs"]
mod __slate_unit_ioctl;
#[path = "genhd.rs"]
mod __slate_unit_genhd;
#[path = "ioprio.rs"]
mod __slate_unit_ioprio;
#[path = "badblocks.rs"]
mod __slate_unit_badblocks;
#[path = "core.rs"]
mod __slate_unit_core;
#[path = "msdos.rs"]
mod __slate_unit_msdos;
#[path = "efi.rs"]
mod __slate_unit_efi;
#[path = "blk_rq_qos.rs"]
mod __slate_unit_blk_rq_qos;
#[path = "disk_events.rs"]
mod __slate_unit_disk_events;
#[path = "blk_ia_ranges.rs"]
mod __slate_unit_blk_ia_ranges;
#[path = "early_lookup.rs"]
mod __slate_unit_early_lookup;
#[path = "bsg.rs"]
mod __slate_unit_bsg;
#[path = "blk_cgroup.rs"]
mod __slate_unit_blk_cgroup;
#[path = "blk_ioprio.rs"]
mod __slate_unit_blk_ioprio;
#[path = "blk_iolatency.rs"]
mod __slate_unit_blk_iolatency;
#[path = "blk_iocost.rs"]
mod __slate_unit_blk_iocost;
#[path = "mq_deadline.rs"]
mod __slate_unit_mq_deadline;
#[path = "kyber_iosched.rs"]
mod __slate_unit_kyber_iosched;
#[path = "blk_mq_debugfs.rs"]
mod __slate_unit_blk_mq_debugfs;
#[path = "blk_pm.rs"]
mod __slate_unit_blk_pm;
#[path = "holder.rs"]
mod __slate_unit_holder;

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

