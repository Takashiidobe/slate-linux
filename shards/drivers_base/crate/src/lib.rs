#![no_std]
#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals, rustdoc::all)]
#![allow(suspicious_runtime_symbol_definitions)]
#![feature(linkage)]
#![feature(link_llvm_intrinsics)]
#![feature(used_with_arg)]

#[path = "component.rs"]
mod __slate_unit_component;
#[path = "core.rs"]
mod __slate_unit_core;
#[path = "bus.rs"]
mod __slate_unit_bus;
#[path = "dd.rs"]
mod __slate_unit_dd;
#[path = "syscore.rs"]
mod __slate_unit_syscore;
#[path = "driver.rs"]
mod __slate_unit_driver;
#[path = "class.rs"]
mod __slate_unit_class;
#[path = "platform.rs"]
mod __slate_unit_platform;
#[path = "cpu.rs"]
mod __slate_unit_cpu;
#[path = "firmware.rs"]
mod __slate_unit_firmware;
#[path = "init.rs"]
mod __slate_unit_init;
#[path = "map.rs"]
mod __slate_unit_map;
#[path = "devres.rs"]
mod __slate_unit_devres;
#[path = "attribute_container.rs"]
mod __slate_unit_attribute_container;
#[path = "transport_class.rs"]
mod __slate_unit_transport_class;
#[path = "topology.rs"]
mod __slate_unit_topology;
#[path = "container.rs"]
mod __slate_unit_container;
#[path = "property.rs"]
mod __slate_unit_property;
#[path = "cacheinfo.rs"]
mod __slate_unit_cacheinfo;
#[path = "swnode.rs"]
mod __slate_unit_swnode;
#[path = "faux.rs"]
mod __slate_unit_faux;
#[path = "auxiliary.rs"]
mod __slate_unit_auxiliary;
#[path = "devtmpfs.rs"]
mod __slate_unit_devtmpfs;
#[path = "sysfs.rs"]
mod __slate_unit_sysfs;
#[path = "generic_ops.rs"]
mod __slate_unit_generic_ops;
#[path = "common.rs"]
mod __slate_unit_common;
#[path = "qos.rs"]
mod __slate_unit_qos;
#[path = "runtime.rs"]
mod __slate_unit_runtime;
#[path = "wakeirq.rs"]
mod __slate_unit_wakeirq;
#[path = "power_main.rs"]
mod __slate_unit_power_main;
#[path = "wakeup.rs"]
mod __slate_unit_wakeup;
#[path = "wakeup_stats.rs"]
mod __slate_unit_wakeup_stats;
#[path = "power_trace.rs"]
mod __slate_unit_power_trace;
#[path = "firmware_loader_main.rs"]
mod __slate_unit_firmware_loader_main;
#[path = "builtin_main.rs"]
mod __slate_unit_builtin_main;
#[path = "node.rs"]
mod __slate_unit_node;
#[path = "module.rs"]
mod __slate_unit_module;
#[path = "auxiliary_sysfs.rs"]
mod __slate_unit_auxiliary_sysfs;
#[path = "regmap.rs"]
mod __slate_unit_regmap;
#[path = "regcache.rs"]
mod __slate_unit_regcache;
#[path = "regcache_rbtree.rs"]
mod __slate_unit_regcache_rbtree;
#[path = "regcache_flat.rs"]
mod __slate_unit_regcache_flat;
#[path = "regcache_maple.rs"]
mod __slate_unit_regcache_maple;
#[path = "regmap_debugfs.rs"]
mod __slate_unit_regmap_debugfs;
#[path = "platform_msi.rs"]
mod __slate_unit_platform_msi;
#[path = "physical_location.rs"]
mod __slate_unit_physical_location;
#[path = "base_trace.rs"]
mod __slate_unit_base_trace;

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

