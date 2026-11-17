#![no_std]
#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals, rustdoc::all)]
#![allow(suspicious_runtime_symbol_definitions)]
#![feature(linkage)]
#![feature(link_llvm_intrinsics)]
#![feature(used_with_arg)]

#[path = "syscall_64.rs"]
mod __slate_unit_syscall_64;
#[path = "entry_common.rs"]
mod __slate_unit_entry_common;
#[path = "vma.rs"]
mod __slate_unit_vma;
#[path = "vdso_extable.rs"]
mod __slate_unit_vdso_extable;
#[path = "vdso32_setup.rs"]
mod __slate_unit_vdso32_setup;
#[path = "vsyscall_64.rs"]
mod __slate_unit_vsyscall_64;
#[path = "syscall_32.rs"]
mod __slate_unit_syscall_32;
#[path = "events_core.rs"]
mod __slate_unit_events_core;
#[path = "probe.rs"]
mod __slate_unit_probe;
#[path = "utils.rs"]
mod __slate_unit_utils;
#[path = "rapl.rs"]
mod __slate_unit_rapl;
#[path = "amd_core.rs"]
mod __slate_unit_amd_core;
#[path = "amd_lbr.rs"]
mod __slate_unit_amd_lbr;
#[path = "ibs.rs"]
mod __slate_unit_ibs;
#[path = "amd_uncore.rs"]
mod __slate_unit_amd_uncore;
#[path = "iommu.rs"]
mod __slate_unit_iommu;
#[path = "events_msr.rs"]
mod __slate_unit_events_msr;
#[path = "intel_core.rs"]
mod __slate_unit_intel_core;
#[path = "bts.rs"]
mod __slate_unit_bts;
#[path = "ds.rs"]
mod __slate_unit_ds;
#[path = "knc.rs"]
mod __slate_unit_knc;
#[path = "intel_lbr.rs"]
mod __slate_unit_intel_lbr;
#[path = "p4.rs"]
mod __slate_unit_p4;
#[path = "p6.rs"]
mod __slate_unit_p6;
#[path = "pt.rs"]
mod __slate_unit_pt;
#[path = "intel_uncore.rs"]
mod __slate_unit_intel_uncore;
#[path = "uncore_nhmex.rs"]
mod __slate_unit_uncore_nhmex;
#[path = "uncore_snb.rs"]
mod __slate_unit_uncore_snb;
#[path = "uncore_snbep.rs"]
mod __slate_unit_uncore_snbep;
#[path = "uncore_discovery.rs"]
mod __slate_unit_uncore_discovery;
#[path = "intel_cstate.rs"]
mod __slate_unit_intel_cstate;
#[path = "zhaoxin_core.rs"]
mod __slate_unit_zhaoxin_core;
#[path = "fpu_init.rs"]
mod __slate_unit_fpu_init;
#[path = "bugs.rs"]
mod __slate_unit_bugs;
#[path = "fpu_core.rs"]
mod __slate_unit_fpu_core;
#[path = "regset.rs"]
mod __slate_unit_regset;
#[path = "signal.rs"]
mod __slate_unit_signal;
#[path = "xstate.rs"]
mod __slate_unit_xstate;
#[path = "boot.rs"]
mod __slate_unit_boot;
#[path = "sleep.rs"]
mod __slate_unit_sleep;
#[path = "cppc.rs"]
mod __slate_unit_cppc;
#[path = "madt_wakeup.rs"]
mod __slate_unit_madt_wakeup;
#[path = "acpi_cstate.rs"]
mod __slate_unit_acpi_cstate;
#[path = "apic.rs"]
mod __slate_unit_apic;
#[path = "apic_common.rs"]
mod __slate_unit_apic_common;
#[path = "apic_noop.rs"]
mod __slate_unit_apic_noop;
#[path = "ipi.rs"]
mod __slate_unit_ipi;
#[path = "vector.rs"]
mod __slate_unit_vector;
#[path = "apic_init.rs"]
mod __slate_unit_apic_init;
#[path = "hw_nmi.rs"]
mod __slate_unit_hw_nmi;
#[path = "io_apic.rs"]
mod __slate_unit_io_apic;
#[path = "msi.rs"]
mod __slate_unit_msi;
#[path = "x2apic_phys.rs"]
mod __slate_unit_x2apic_phys;
#[path = "x2apic_cluster.rs"]
mod __slate_unit_x2apic_cluster;
#[path = "apic_flat_64.rs"]
mod __slate_unit_apic_flat_64;
#[path = "probe_64.rs"]
mod __slate_unit_probe_64;
#[path = "kprobes_core.rs"]
mod __slate_unit_kprobes_core;
#[path = "opt.rs"]
mod __slate_unit_opt;
#[path = "mm_init.rs"]
mod __slate_unit_mm_init;
#[path = "init_64.rs"]
mod __slate_unit_init_64;
#[path = "fault.rs"]
mod __slate_unit_fault;
#[path = "ioremap.rs"]
mod __slate_unit_ioremap;
#[path = "mm_extable.rs"]
mod __slate_unit_mm_extable;
#[path = "mmap.rs"]
mod __slate_unit_mmap;
#[path = "pgtable.rs"]
mod __slate_unit_pgtable;
#[path = "physaddr.rs"]
mod __slate_unit_physaddr;
#[path = "tlb.rs"]
mod __slate_unit_tlb;
#[path = "cpu_entry_area.rs"]
mod __slate_unit_cpu_entry_area;
#[path = "maccess.rs"]
mod __slate_unit_maccess;
#[path = "pgprot.rs"]
mod __slate_unit_pgprot;
#[path = "set_memory.rs"]
mod __slate_unit_set_memory;
#[path = "memtype.rs"]
mod __slate_unit_memtype;
#[path = "memtype_interval.rs"]
mod __slate_unit_memtype_interval;
#[path = "hugetlbpage.rs"]
mod __slate_unit_hugetlbpage;
#[path = "dump_pagetables.rs"]
mod __slate_unit_dump_pagetables;
#[path = "numa.rs"]
mod __slate_unit_numa;
#[path = "amdtopology.rs"]
mod __slate_unit_amdtopology;
#[path = "srat.rs"]
mod __slate_unit_srat;
#[path = "pkeys.rs"]
mod __slate_unit_pkeys;
#[path = "mm_kaslr.rs"]
mod __slate_unit_mm_kaslr;
#[path = "pti.rs"]
mod __slate_unit_pti;
#[path = "audit.rs"]
mod __slate_unit_audit;
#[path = "memmap.rs"]
mod __slate_unit_memmap;
#[path = "quirks.rs"]
mod __slate_unit_quirks;
#[path = "efi.rs"]
mod __slate_unit_efi;
#[path = "efi_64.rs"]
mod __slate_unit_efi_64;
#[path = "runtime_map.rs"]
mod __slate_unit_runtime_map;
#[path = "iosf_mbi.rs"]
mod __slate_unit_iosf_mbi;
#[path = "msr_smp.rs"]
mod __slate_unit_msr_smp;
#[path = "cache_smp.rs"]
mod __slate_unit_cache_smp;
#[path = "lib_msr.rs"]
mod __slate_unit_lib_msr;
#[path = "msr_reg_export.rs"]
mod __slate_unit_msr_reg_export;
#[path = "iomem.rs"]
mod __slate_unit_iomem;
#[path = "i386.rs"]
mod __slate_unit_i386;
#[path = "pci_init.rs"]
mod __slate_unit_pci_init;
#[path = "mmconfig_64.rs"]
mod __slate_unit_mmconfig_64;
#[path = "direct.rs"]
mod __slate_unit_direct;
#[path = "mmconfig_shared.rs"]
mod __slate_unit_mmconfig_shared;
#[path = "fixup.rs"]
mod __slate_unit_fixup;
#[path = "acpi.rs"]
mod __slate_unit_acpi;
#[path = "legacy.rs"]
mod __slate_unit_legacy;
#[path = "irq.rs"]
mod __slate_unit_irq;
#[path = "pci_common.rs"]
mod __slate_unit_pci_common;
#[path = "early.rs"]
mod __slate_unit_early;
#[path = "bus_numa.rs"]
mod __slate_unit_bus_numa;
#[path = "amd_bus.rs"]
mod __slate_unit_amd_bus;
#[path = "cmdline.rs"]
mod __slate_unit_cmdline;
#[path = "copy_mc.rs"]
mod __slate_unit_copy_mc;
#[path = "cpu.rs"]
mod __slate_unit_cpu;
#[path = "csum_partial_64.rs"]
mod __slate_unit_csum_partial_64;
#[path = "csum_wrappers_64.rs"]
mod __slate_unit_csum_wrappers_64;
#[path = "delay.rs"]
mod __slate_unit_delay;
#[path = "inat.rs"]
mod __slate_unit_inat;
#[path = "insn_eval.rs"]
mod __slate_unit_insn_eval;
#[path = "insn.rs"]
mod __slate_unit_insn;
#[path = "lib_kaslr.rs"]
mod __slate_unit_lib_kaslr;
#[path = "misc.rs"]
mod __slate_unit_misc;
#[path = "pc_conf_reg.rs"]
mod __slate_unit_pc_conf_reg;
#[path = "usercopy.rs"]
mod __slate_unit_usercopy;
#[path = "usercopy_64.rs"]
mod __slate_unit_usercopy_64;

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

