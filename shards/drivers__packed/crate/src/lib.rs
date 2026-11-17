#![no_std]
#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals, rustdoc::all)]
#![allow(suspicious_runtime_symbol_definitions)]
#![feature(linkage)]
#![feature(link_llvm_intrinsics)]
#![feature(used_with_arg)]

#[path = "srat.rs"]
mod __slate_unit_srat;
#[path = "apple.rs"]
mod __slate_unit_apple;
#[path = "cmos_rtc.rs"]
mod __slate_unit_cmos_rtc;
#[path = "lpss.rs"]
mod __slate_unit_lpss;
#[path = "s2idle.rs"]
mod __slate_unit_s2idle;
#[path = "utils.rs"]
mod __slate_unit_utils;
#[path = "blacklist.rs"]
mod __slate_unit_blacklist;
#[path = "dmaengine.rs"]
mod __slate_unit_dmaengine;
#[path = "virt_dma.rs"]
mod __slate_unit_virt_dma;
#[path = "acpi_dma.rs"]
mod __slate_unit_acpi_dma;
#[path = "dw_core.rs"]
mod __slate_unit_dw_core;
#[path = "dw.rs"]
mod __slate_unit_dw;
#[path = "idma32.rs"]
mod __slate_unit_idma32;
#[path = "acpi.rs"]
mod __slate_unit_acpi;
#[path = "hsu.rs"]
mod __slate_unit_hsu;
#[path = "mem.rs"]
mod __slate_unit_mem;
#[path = "random.rs"]
mod __slate_unit_random;
#[path = "misc.rs"]
mod __slate_unit_misc;
#[path = "virtio_console.rs"]
mod __slate_unit_virtio_console;
#[path = "hpet.rs"]
mod __slate_unit_hpet;
#[path = "nvram.rs"]
mod __slate_unit_nvram;
#[path = "hw_random_core.rs"]
mod __slate_unit_hw_random_core;
#[path = "via_rng.rs"]
mod __slate_unit_via_rng;
#[path = "backend.rs"]
mod __slate_unit_backend;
#[path = "generic.rs"]
mod __slate_unit_generic;
#[path = "isoch.rs"]
mod __slate_unit_isoch;
#[path = "amd64_agp.rs"]
mod __slate_unit_amd64_agp;
#[path = "intel_agp.rs"]
mod __slate_unit_intel_agp;
#[path = "intel_gtt.rs"]
mod __slate_unit_intel_gtt;
#[path = "buddy.rs"]
mod __slate_unit_buddy;
#[path = "cn_queue.rs"]
mod __slate_unit_cn_queue;
#[path = "connector.rs"]
mod __slate_unit_connector;
#[path = "cn_proc.rs"]
mod __slate_unit_cn_proc;
#[path = "loop.rs"]
mod __slate_unit_loop;
#[path = "virtio_blk.rs"]
mod __slate_unit_virtio_blk;
#[path = "dma_buf.rs"]
mod __slate_unit_dma_buf;
#[path = "dma_fence.rs"]
mod __slate_unit_dma_fence;
#[path = "dma_fence_array.rs"]
mod __slate_unit_dma_fence_array;
#[path = "dma_fence_chain.rs"]
mod __slate_unit_dma_fence_chain;
#[path = "dma_fence_unwrap.rs"]
mod __slate_unit_dma_fence_unwrap;
#[path = "dma_resv.rs"]
mod __slate_unit_dma_resv;
#[path = "dma_buf_mapping.rs"]
mod __slate_unit_dma_buf_mapping;
#[path = "sync_file.rs"]
mod __slate_unit_sync_file;
#[path = "libata_core.rs"]
mod __slate_unit_libata_core;
#[path = "libata_scsi.rs"]
mod __slate_unit_libata_scsi;
#[path = "libata_eh.rs"]
mod __slate_unit_libata_eh;
#[path = "libata_transport.rs"]
mod __slate_unit_libata_transport;
#[path = "libata_trace.rs"]
mod __slate_unit_libata_trace;
#[path = "libata_sata.rs"]
mod __slate_unit_libata_sata;
#[path = "libata_sff.rs"]
mod __slate_unit_libata_sff;
#[path = "libata_pmp.rs"]
mod __slate_unit_libata_pmp;
#[path = "libata_acpi.rs"]
mod __slate_unit_libata_acpi;
#[path = "libata_pata_timings.rs"]
mod __slate_unit_libata_pata_timings;
#[path = "ahci.rs"]
mod __slate_unit_ahci;
#[path = "libahci.rs"]
mod __slate_unit_libahci;
#[path = "ata_piix.rs"]
mod __slate_unit_ata_piix;
#[path = "pata_amd.rs"]
mod __slate_unit_pata_amd;
#[path = "pata_oldpiix.rs"]
mod __slate_unit_pata_oldpiix;
#[path = "pata_sch.rs"]
mod __slate_unit_pata_sch;
#[path = "init_ohci1394_dma.rs"]
mod __slate_unit_init_ohci1394_dma;
#[path = "cdrom.rs"]
mod __slate_unit_cdrom;
#[path = "i2c_boardinfo.rs"]
mod __slate_unit_i2c_boardinfo;
#[path = "i2c_core_base.rs"]
mod __slate_unit_i2c_core_base;
#[path = "i2c_core_smbus.rs"]
mod __slate_unit_i2c_core_smbus;
#[path = "i2c_core_acpi.rs"]
mod __slate_unit_i2c_core_acpi;
#[path = "i2c_smbus.rs"]
mod __slate_unit_i2c_smbus;
#[path = "i2c_algo_bit.rs"]
mod __slate_unit_i2c_algo_bit;
#[path = "i2c_i801.rs"]
mod __slate_unit_i2c_i801;
#[path = "hwmon.rs"]
mod __slate_unit_hwmon;
#[path = "cpufreq.rs"]
mod __slate_unit_cpufreq;
#[path = "freq_table.rs"]
mod __slate_unit_freq_table;
#[path = "cpufreq_performance.rs"]
mod __slate_unit_cpufreq_performance;
#[path = "cpufreq_userspace.rs"]
mod __slate_unit_cpufreq_userspace;
#[path = "cpufreq_ondemand.rs"]
mod __slate_unit_cpufreq_ondemand;
#[path = "cpufreq_governor.rs"]
mod __slate_unit_cpufreq_governor;
#[path = "cpufreq_governor_attr_set.rs"]
mod __slate_unit_cpufreq_governor_attr_set;
#[path = "acpi_cpufreq.rs"]
mod __slate_unit_acpi_cpufreq;
#[path = "amd_pstate.rs"]
mod __slate_unit_amd_pstate;
#[path = "amd_pstate_trace.rs"]
mod __slate_unit_amd_pstate_trace;
#[path = "intel_pstate.rs"]
mod __slate_unit_intel_pstate;
#[path = "cpuidle.rs"]
mod __slate_unit_cpuidle;
#[path = "driver.rs"]
mod __slate_unit_driver;
#[path = "governor.rs"]
mod __slate_unit_governor;
#[path = "sysfs.rs"]
mod __slate_unit_sysfs;
#[path = "menu.rs"]
mod __slate_unit_menu;
#[path = "haltpoll.rs"]
mod __slate_unit_haltpoll;
#[path = "poll_state.rs"]
mod __slate_unit_poll_state;
#[path = "cpuidle_haltpoll.rs"]
mod __slate_unit_cpuidle_haltpoll;
#[path = "dmi_scan.rs"]
mod __slate_unit_dmi_scan;
#[path = "dmi_id.rs"]
mod __slate_unit_dmi_id;
#[path = "firmware_memmap.rs"]
mod __slate_unit_firmware_memmap;
#[path = "efi_bgrt.rs"]
mod __slate_unit_efi_bgrt;
#[path = "efi.rs"]
mod __slate_unit_efi;
#[path = "vars.rs"]
mod __slate_unit_vars;
#[path = "reboot.rs"]
mod __slate_unit_reboot;
#[path = "memattr.rs"]
mod __slate_unit_memattr;
#[path = "tpm.rs"]
mod __slate_unit_tpm;
#[path = "efi_memmap.rs"]
mod __slate_unit_efi_memmap;
#[path = "esrt.rs"]
mod __slate_unit_esrt;
#[path = "runtime_wrappers.rs"]
mod __slate_unit_runtime_wrappers;
#[path = "earlycon.rs"]
mod __slate_unit_earlycon;
#[path = "acpi_pm.rs"]
mod __slate_unit_acpi_pm;
#[path = "i8253.rs"]
mod __slate_unit_i8253;
#[path = "hid_hid_core.rs"]
mod __slate_unit_hid_hid_core;
#[path = "hid_input.rs"]
mod __slate_unit_hid_input;
#[path = "hid_quirks.rs"]
mod __slate_unit_hid_quirks;
#[path = "hid_debug.rs"]
mod __slate_unit_hid_debug;
#[path = "hidraw.rs"]
mod __slate_unit_hidraw;
#[path = "hid_generic.rs"]
mod __slate_unit_hid_generic;
#[path = "hid_gyration.rs"]
mod __slate_unit_hid_gyration;
#[path = "hid_ntrig.rs"]
mod __slate_unit_hid_ntrig;
#[path = "hid_pl.rs"]
mod __slate_unit_hid_pl;
#[path = "hid_petalynx.rs"]
mod __slate_unit_hid_petalynx;
#[path = "hid_samsung.rs"]
mod __slate_unit_hid_samsung;
#[path = "hid_sony.rs"]
mod __slate_unit_hid_sony;
#[path = "hid_sunplus.rs"]
mod __slate_unit_hid_sunplus;
#[path = "hid_topseed.rs"]
mod __slate_unit_hid_topseed;
#[path = "usbhid_hid_core.rs"]
mod __slate_unit_usbhid_hid_core;
#[path = "hiddev.rs"]
mod __slate_unit_hiddev;
#[path = "hid_pidff.rs"]
mod __slate_unit_hid_pidff;

mod __slate_panic {
    #[panic_handler]
    fn __slate_panic_handler(_: &core::panic::PanicInfo) -> ! {
        loop {}
    }

    #[linkage = "weak"]
    #[unsafe(no_mangle)]
    extern "C" fn rust_eh_personality() {}
}

