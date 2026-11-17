#![no_std]
#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals, rustdoc::all)]
#![allow(suspicious_runtime_symbol_definitions)]
#![feature(linkage)]
#![feature(link_llvm_intrinsics)]
#![feature(used_with_arg)]

#[path = "vdso64_image.rs"]
mod __slate_unit_vdso64_image;
#[path = "vdso32_image.rs"]
mod __slate_unit_vdso32_image;
#[path = "realmode_init.rs"]
mod __slate_unit_realmode_init;
#[path = "capflags.rs"]
mod __slate_unit_capflags;
#[path = "cmdline.rs"]
mod __slate_unit_cmdline;
#[path = "system_keyring.rs"]
mod __slate_unit_system_keyring;
#[path = "rsapubkey_asn1.rs"]
mod __slate_unit_rsapubkey_asn1;
#[path = "rsaprivkey_asn1.rs"]
mod __slate_unit_rsaprivkey_asn1;
#[path = "x509_asn1.rs"]
mod __slate_unit_x509_asn1;
#[path = "x509_akid_asn1.rs"]
mod __slate_unit_x509_akid_asn1;
#[path = "pkcs7_asn1.rs"]
mod __slate_unit_pkcs7_asn1;
#[path = "irq_msi_lib.rs"]
mod __slate_unit_irq_msi_lib;
#[path = "led_core.rs"]
mod __slate_unit_led_core;
#[path = "led_class.rs"]
mod __slate_unit_led_class;
#[path = "led_triggers.rs"]
mod __slate_unit_led_triggers;
#[path = "defkeymap.rs"]
mod __slate_unit_defkeymap;
#[path = "consolemap_deftbl.rs"]
mod __slate_unit_consolemap_deftbl;
#[path = "amd_iommu.rs"]
mod __slate_unit_amd_iommu;
#[path = "amd_init.rs"]
mod __slate_unit_amd_init;
#[path = "quirks.rs"]
mod __slate_unit_quirks;
#[path = "ppr.rs"]
mod __slate_unit_ppr;
#[path = "amd_pasid.rs"]
mod __slate_unit_amd_pasid;
#[path = "intel_iommu.rs"]
mod __slate_unit_intel_iommu;
#[path = "intel_pasid.rs"]
mod __slate_unit_intel_pasid;
#[path = "nested.rs"]
mod __slate_unit_nested;
#[path = "cache.rs"]
mod __slate_unit_cache;
#[path = "prq.rs"]
mod __slate_unit_prq;
#[path = "dmar.rs"]
mod __slate_unit_dmar;
#[path = "trace.rs"]
mod __slate_unit_trace;
#[path = "intel_irq_remapping.rs"]
mod __slate_unit_intel_irq_remapping;
#[path = "perfmon.rs"]
mod __slate_unit_perfmon;
#[path = "iommu_amdv1.rs"]
mod __slate_unit_iommu_amdv1;
#[path = "iommu_vtdss.rs"]
mod __slate_unit_iommu_vtdss;
#[path = "iommu_x86_64.rs"]
mod __slate_unit_iommu_x86_64;
#[path = "iommu_iommu.rs"]
mod __slate_unit_iommu_iommu;
#[path = "iommu_pages.rs"]
mod __slate_unit_iommu_pages;
#[path = "iommu_traces.rs"]
mod __slate_unit_iommu_traces;
#[path = "iommu_sysfs.rs"]
mod __slate_unit_iommu_sysfs;
#[path = "dma_iommu.rs"]
mod __slate_unit_dma_iommu;
#[path = "iova.rs"]
mod __slate_unit_iova;
#[path = "iommu_irq_remapping.rs"]
mod __slate_unit_iommu_irq_remapping;
#[path = "iommu_sva.rs"]
mod __slate_unit_iommu_sva;
#[path = "io_pgfault.rs"]
mod __slate_unit_io_pgfault;
#[path = "eeprom_93cx6.rs"]
mod __slate_unit_eeprom_93cx6;
#[path = "mei_init.rs"]
mod __slate_unit_mei_init;
#[path = "hbm.rs"]
mod __slate_unit_hbm;
#[path = "interrupt.rs"]
mod __slate_unit_interrupt;
#[path = "client.rs"]
mod __slate_unit_client;
#[path = "__slate_unit_main.rs"]
mod __slate_unit_main;
#[path = "dma_ring.rs"]
mod __slate_unit_dma_ring;
#[path = "bus.rs"]
mod __slate_unit_bus;
#[path = "bus_fixup.rs"]
mod __slate_unit_bus_fixup;
#[path = "debugfs.rs"]
mod __slate_unit_debugfs;
#[path = "mei_trace.rs"]
mod __slate_unit_mei_trace;
#[path = "pci_me.rs"]
mod __slate_unit_pci_me;
#[path = "hw_me.rs"]
mod __slate_unit_hw_me;
#[path = "mac_hid.rs"]
mod __slate_unit_mac_hid;
#[path = "serio.rs"]
mod __slate_unit_serio;
#[path = "i8042.rs"]
mod __slate_unit_i8042;
#[path = "serport.rs"]
mod __slate_unit_serport;
#[path = "libps2.rs"]
mod __slate_unit_libps2;
#[path = "input.rs"]
mod __slate_unit_input;
#[path = "input_compat.rs"]
mod __slate_unit_input_compat;
#[path = "input_mt.rs"]
mod __slate_unit_input_mt;
#[path = "input_poller.rs"]
mod __slate_unit_input_poller;
#[path = "ff_core.rs"]
mod __slate_unit_ff_core;
#[path = "touchscreen.rs"]
mod __slate_unit_touchscreen;
#[path = "touch_overlay.rs"]
mod __slate_unit_touch_overlay;
#[path = "ff_memless.rs"]
mod __slate_unit_ff_memless;
#[path = "sparse_keymap.rs"]
mod __slate_unit_sparse_keymap;
#[path = "vivaldi_fmap.rs"]
mod __slate_unit_vivaldi_fmap;
#[path = "input_leds.rs"]
mod __slate_unit_input_leds;
#[path = "evdev.rs"]
mod __slate_unit_evdev;
#[path = "atkbd.rs"]
mod __slate_unit_atkbd;
#[path = "psmouse_base.rs"]
mod __slate_unit_psmouse_base;
#[path = "synaptics.rs"]
mod __slate_unit_synaptics;
#[path = "focaltech.rs"]
mod __slate_unit_focaltech;
#[path = "alps.rs"]
mod __slate_unit_alps;
#[path = "byd.rs"]
mod __slate_unit_byd;
#[path = "logips2pp.rs"]
mod __slate_unit_logips2pp;
#[path = "lifebook.rs"]
mod __slate_unit_lifebook;
#[path = "trackpoint.rs"]
mod __slate_unit_trackpoint;
#[path = "cypress_ps2.rs"]
mod __slate_unit_cypress_ps2;
#[path = "psmouse_smbus.rs"]
mod __slate_unit_psmouse_smbus;
#[path = "md.rs"]
mod __slate_unit_md;
#[path = "md_bitmap.rs"]
mod __slate_unit_md_bitmap;
#[path = "md_autodetect.rs"]
mod __slate_unit_md_autodetect;
#[path = "dm.rs"]
mod __slate_unit_dm;
#[path = "dm_table.rs"]
mod __slate_unit_dm_table;
#[path = "dm_target.rs"]
mod __slate_unit_dm_target;
#[path = "dm_linear.rs"]
mod __slate_unit_dm_linear;
#[path = "dm_stripe.rs"]
mod __slate_unit_dm_stripe;
#[path = "dm_ioctl.rs"]
mod __slate_unit_dm_ioctl;
#[path = "dm_io.rs"]
mod __slate_unit_dm_io;
#[path = "dm_kcopyd.rs"]
mod __slate_unit_dm_kcopyd;
#[path = "dm_sysfs.rs"]
mod __slate_unit_dm_sysfs;
#[path = "dm_stats.rs"]
mod __slate_unit_dm_stats;
#[path = "dm_rq.rs"]
mod __slate_unit_dm_rq;
#[path = "dm_io_rewind.rs"]
mod __slate_unit_dm_io_rewind;
#[path = "dm_builtin.rs"]
mod __slate_unit_dm_builtin;
#[path = "dm_raid1.rs"]
mod __slate_unit_dm_raid1;
#[path = "dm_log.rs"]
mod __slate_unit_dm_log;
#[path = "dm_region_hash.rs"]
mod __slate_unit_dm_region_hash;
#[path = "dm_zero.rs"]
mod __slate_unit_dm_zero;
#[path = "wmi_bmof.rs"]
mod __slate_unit_wmi_bmof;
#[path = "eeepc_laptop.rs"]
mod __slate_unit_eeepc_laptop;
#[path = "p2sb.rs"]
mod __slate_unit_p2sb;
#[path = "wmi_core.rs"]
mod __slate_unit_wmi_core;
#[path = "marshalling.rs"]
mod __slate_unit_marshalling;
#[path = "string.rs"]
mod __slate_unit_string;
#[path = "mailbox.rs"]
mod __slate_unit_mailbox;
#[path = "pcc.rs"]
mod __slate_unit_pcc;
#[path = "nvmem_core.rs"]
mod __slate_unit_nvmem_core;
#[path = "shipped_certs.rs"]
mod __slate_unit_shipped_certs;
#[path = "cpu.rs"]
mod __slate_unit_cpu;
#[path = "hibernate_64.rs"]
mod __slate_unit_hibernate_64;
#[path = "hibernate.rs"]
mod __slate_unit_hibernate;
#[path = "video_common.rs"]
mod __slate_unit_video_common;

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

