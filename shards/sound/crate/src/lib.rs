#![no_std]
#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals, rustdoc::all)]
#![allow(suspicious_runtime_symbol_definitions)]
#![feature(asm_goto_with_outputs)]
#![feature(linkage)]
#![feature(link_llvm_intrinsics)]
#![feature(used_with_arg)]

#[path = "sound_core.rs"]
mod __slate_unit_sound_core;
#[path = "sound.rs"]
mod __slate_unit_sound;
#[path = "init.rs"]
mod __slate_unit_init;
#[path = "memory.rs"]
mod __slate_unit_memory;
#[path = "control.rs"]
mod __slate_unit_control;
#[path = "misc.rs"]
mod __slate_unit_misc;
#[path = "sound_core_device.rs"]
mod __slate_unit_sound_core_device;
#[path = "info.rs"]
mod __slate_unit_info;
#[path = "isadma.rs"]
mod __slate_unit_isadma;
#[path = "vmaster.rs"]
mod __slate_unit_vmaster;
#[path = "ctljack.rs"]
mod __slate_unit_ctljack;
#[path = "core_jack.rs"]
mod __slate_unit_core_jack;
#[path = "core_hwdep.rs"]
mod __slate_unit_core_hwdep;
#[path = "timer.rs"]
mod __slate_unit_timer;
#[path = "hrtimer.rs"]
mod __slate_unit_hrtimer;
#[path = "pcm.rs"]
mod __slate_unit_pcm;
#[path = "pcm_native.rs"]
mod __slate_unit_pcm_native;
#[path = "pcm_lib.rs"]
mod __slate_unit_pcm_lib;
#[path = "pcm_misc.rs"]
mod __slate_unit_pcm_misc;
#[path = "pcm_memory.rs"]
mod __slate_unit_pcm_memory;
#[path = "memalloc.rs"]
mod __slate_unit_memalloc;
#[path = "pcm_timer.rs"]
mod __slate_unit_pcm_timer;
#[path = "seq_device.rs"]
mod __slate_unit_seq_device;
#[path = "seq.rs"]
mod __slate_unit_seq;
#[path = "seq_lock.rs"]
mod __slate_unit_seq_lock;
#[path = "seq_clientmgr.rs"]
mod __slate_unit_seq_clientmgr;
#[path = "seq_memory.rs"]
mod __slate_unit_seq_memory;
#[path = "seq_queue.rs"]
mod __slate_unit_seq_queue;
#[path = "seq_fifo.rs"]
mod __slate_unit_seq_fifo;
#[path = "seq_prioq.rs"]
mod __slate_unit_seq_prioq;
#[path = "seq_timer.rs"]
mod __slate_unit_seq_timer;
#[path = "seq_system.rs"]
mod __slate_unit_seq_system;
#[path = "seq_ports.rs"]
mod __slate_unit_seq_ports;
#[path = "seq_info.rs"]
mod __slate_unit_seq_info;
#[path = "seq_dummy.rs"]
mod __slate_unit_seq_dummy;
#[path = "hda_bus_type.rs"]
mod __slate_unit_hda_bus_type;
#[path = "bus.rs"]
mod __slate_unit_bus;
#[path = "hda_core_device.rs"]
mod __slate_unit_hda_core_device;
#[path = "core_sysfs.rs"]
mod __slate_unit_core_sysfs;
#[path = "regmap.rs"]
mod __slate_unit_regmap;
#[path = "core_controller.rs"]
mod __slate_unit_core_controller;
#[path = "stream.rs"]
mod __slate_unit_stream;
#[path = "array.rs"]
mod __slate_unit_array;
#[path = "hdmi_chmap.rs"]
mod __slate_unit_hdmi_chmap;
#[path = "trace.rs"]
mod __slate_unit_trace;
#[path = "component.rs"]
mod __slate_unit_component;
#[path = "i915.rs"]
mod __slate_unit_i915;
#[path = "intel_dsp_config.rs"]
mod __slate_unit_intel_dsp_config;
#[path = "intel_nhlt.rs"]
mod __slate_unit_intel_nhlt;
#[path = "intel_sdw_acpi.rs"]
mod __slate_unit_intel_sdw_acpi;
#[path = "bind.rs"]
mod __slate_unit_bind;
#[path = "codec.rs"]
mod __slate_unit_codec;
#[path = "common_jack.rs"]
mod __slate_unit_common_jack;
#[path = "auto_parser.rs"]
mod __slate_unit_auto_parser;
#[path = "common_sysfs.rs"]
mod __slate_unit_common_sysfs;
#[path = "common_controller.rs"]
mod __slate_unit_common_controller;
#[path = "proc.rs"]
mod __slate_unit_proc;
#[path = "common_hwdep.rs"]
mod __slate_unit_common_hwdep;
#[path = "intel.rs"]
mod __slate_unit_intel;
#[path = "last.rs"]
mod __slate_unit_last;

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

