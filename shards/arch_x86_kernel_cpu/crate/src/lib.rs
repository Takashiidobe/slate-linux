#![no_std]
#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals, rustdoc::all)]
#![allow(suspicious_runtime_symbol_definitions)]
#![feature(linkage)]
#![feature(link_llvm_intrinsics)]
#![feature(used_with_arg)]

#[path = "cacheinfo.rs"]
mod __slate_unit_cacheinfo;
#[path = "scattered.rs"]
mod __slate_unit_scattered;
#[path = "topology_common.rs"]
mod __slate_unit_topology_common;
#[path = "topology_ext.rs"]
mod __slate_unit_topology_ext;
#[path = "topology_amd.rs"]
mod __slate_unit_topology_amd;
#[path = "cpuid_parser.rs"]
mod __slate_unit_cpuid_parser;
#[path = "common.rs"]
mod __slate_unit_common;
#[path = "rdrand.rs"]
mod __slate_unit_rdrand;
#[path = "match.rs"]
mod __slate_unit_match;
#[path = "bugs.rs"]
mod __slate_unit_bugs;
#[path = "aperfmperf.rs"]
mod __slate_unit_aperfmperf;
#[path = "cpuid_deps.rs"]
mod __slate_unit_cpuid_deps;
#[path = "cpuid_0x2_table.rs"]
mod __slate_unit_cpuid_0x2_table;
#[path = "umwait.rs"]
mod __slate_unit_umwait;
#[path = "powerflags.rs"]
mod __slate_unit_powerflags;
#[path = "topology.rs"]
mod __slate_unit_topology;
#[path = "proc.rs"]
mod __slate_unit_proc;
#[path = "feat_ctl.rs"]
mod __slate_unit_feat_ctl;
#[path = "cpu_intel.rs"]
mod __slate_unit_cpu_intel;
#[path = "tsx.rs"]
mod __slate_unit_tsx;
#[path = "intel_epb.rs"]
mod __slate_unit_intel_epb;
#[path = "cpu_amd.rs"]
mod __slate_unit_cpu_amd;
#[path = "amd_cache_disable.rs"]
mod __slate_unit_amd_cache_disable;
#[path = "hygon.rs"]
mod __slate_unit_hygon;
#[path = "centaur.rs"]
mod __slate_unit_centaur;
#[path = "zhaoxin.rs"]
mod __slate_unit_zhaoxin;
#[path = "mce_core.rs"]
mod __slate_unit_mce_core;
#[path = "severity.rs"]
mod __slate_unit_severity;
#[path = "genpool.rs"]
mod __slate_unit_genpool;
#[path = "mce_intel.rs"]
mod __slate_unit_mce_intel;
#[path = "mce_amd.rs"]
mod __slate_unit_mce_amd;
#[path = "threshold.rs"]
mod __slate_unit_threshold;
#[path = "mtrr.rs"]
mod __slate_unit_mtrr;
#[path = "if.rs"]
mod __slate_unit_if;
#[path = "generic.rs"]
mod __slate_unit_generic;
#[path = "cleanup.rs"]
mod __slate_unit_cleanup;
#[path = "microcode_core.rs"]
mod __slate_unit_microcode_core;
#[path = "microcode_intel.rs"]
mod __slate_unit_microcode_intel;
#[path = "microcode_amd.rs"]
mod __slate_unit_microcode_amd;
#[path = "perfctr_watchdog.rs"]
mod __slate_unit_perfctr_watchdog;
#[path = "vmware.rs"]
mod __slate_unit_vmware;
#[path = "hypervisor.rs"]
mod __slate_unit_hypervisor;
#[path = "mshyperv.rs"]
mod __slate_unit_mshyperv;
#[path = "debugfs.rs"]
mod __slate_unit_debugfs;
#[path = "bus_lock.rs"]
mod __slate_unit_bus_lock;

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

